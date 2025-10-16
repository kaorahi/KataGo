#include <jni.h>
#include <android/log.h>
#include <string>
#include <sstream>
#include <map>
#include <memory>
#include <chrono>
#include <thread>


#include "../core/global.h"
#include "../core/commandloop.h"
#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../core/timer.h"
#include "../core/datetime.h"
#include "../core/makedir.h"
#include "../dataio/sgf.h"
#include "../search/searchnode.h"
#include "../search/asyncbot.h"
#include "../search/patternbonustable.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../main.h"

using namespace std;

inline void testAssert(bool cond) {}

/*
  Based on gtp.cpp
*/
//Assumes that stones are worth 15 points area and 14 points territory, and that 7 komi is fair
static double initialBlackAdvantage(const BoardHistory& hist) {
  BoardHistory histCopy = hist;
  histCopy.setAssumeMultipleStartingBlackMovesAreHandicap(true);
  int handicapStones = histCopy.computeNumHandicapStones();
  if(handicapStones <= 1)
    return 7.0 - hist.rules.komi;

  //Subtract one since white gets the first move afterward
  int extraBlackStones = handicapStones - 1;
  double stoneValue = hist.rules.scoringRule == Rules::SCORING_AREA ? 15.0 : 14.0;
  double whiteHandicapBonus = 0.0;
  if(hist.rules.whiteHandicapBonusRule == Rules::WHB_N)
    whiteHandicapBonus += handicapStones;
  else if(hist.rules.whiteHandicapBonusRule == Rules::WHB_N_MINUS_ONE)
    whiteHandicapBonus += handicapStones-1;

  return stoneValue * extraBlackStones + (7.0 - hist.rules.komi - whiteHandicapBonus);
}

static double getBoardSizeScaling(const Board& board) {
  return pow(19.0 * 19.0 / (double)(board.x_size * board.y_size), 0.75);
}
static double getPointsThresholdForHandicapGame(double boardSizeScaling) {
  return std::max(4.0 / boardSizeScaling, 2.0);
}

static bool noWhiteStonesOnBoard(const Board& board) {
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(board.colors[loc] == P_WHITE)
        return false;
    }
  }
  return true;
}

static void updateDynamicPDAHelper(
  const Board& board, const BoardHistory& hist,
  const double dynamicPlayoutDoublingAdvantageCapPerOppLead,
  const vector<double>& recentWinLossValues,
  double& desiredDynamicPDAForWhite
) {
  (void)board;
  if(dynamicPlayoutDoublingAdvantageCapPerOppLead <= 0.0) {
    desiredDynamicPDAForWhite = 0.0;
  }
  else {
    double boardSizeScaling = getBoardSizeScaling(board);
    double pdaScalingStartPoints = getPointsThresholdForHandicapGame(boardSizeScaling);
    double initialBlackAdvantageInPoints = initialBlackAdvantage(hist);
    Player disadvantagedPla = initialBlackAdvantageInPoints >= 0 ? P_WHITE : P_BLACK;
    double initialAdvantageInPoints = std::fabs(initialBlackAdvantageInPoints);
    if(initialAdvantageInPoints < pdaScalingStartPoints || board.x_size <= 7 || board.y_size <= 7) {
      desiredDynamicPDAForWhite = 0.0;
    }
    else {
      double desiredDynamicPDAForDisadvantagedPla =
        (disadvantagedPla == P_WHITE) ? desiredDynamicPDAForWhite : -desiredDynamicPDAForWhite;

      //What increment to adjust desiredPDA at.
      //Power of 2 to avoid any rounding issues.
      const double increment = 0.125;

      //Hard cap of 2.75 in this parameter, since more extreme values start to reach into values without good training.
      //Scale mildly with board size - small board a given point lead counts as "more".
      double pdaCap = std::min(
        2.75,
        dynamicPlayoutDoublingAdvantageCapPerOppLead *
        (initialAdvantageInPoints - pdaScalingStartPoints) * boardSizeScaling
      );
      pdaCap = round(pdaCap / increment) * increment;

      //No history, or literally no white stones on board? Then this is a new game or a newly set position
      if(recentWinLossValues.size() <= 0 || noWhiteStonesOnBoard(board)) {
        //Just use the cap
        desiredDynamicPDAForDisadvantagedPla = pdaCap;
      }
      else {
        double winLossValue = recentWinLossValues[recentWinLossValues.size()-1];
        //Convert to perspective of disadvantagedPla
        if(disadvantagedPla == P_BLACK)
          winLossValue = -winLossValue;

        //Keep winLossValue between 5% and 25%, subject to available caps.
        if(winLossValue < -0.9)
          desiredDynamicPDAForDisadvantagedPla = desiredDynamicPDAForDisadvantagedPla + 0.125;
        else if(winLossValue > -0.5)
          desiredDynamicPDAForDisadvantagedPla = desiredDynamicPDAForDisadvantagedPla - 0.125;

        desiredDynamicPDAForDisadvantagedPla = std::max(desiredDynamicPDAForDisadvantagedPla, 0.0);
        desiredDynamicPDAForDisadvantagedPla = std::min(desiredDynamicPDAForDisadvantagedPla, pdaCap);
      }

      desiredDynamicPDAForWhite = (disadvantagedPla == P_WHITE) ? desiredDynamicPDAForDisadvantagedPla : -desiredDynamicPDAForDisadvantagedPla;
    }
  }
}

struct ContextConfig {
  enabled_t friendlyPass;
  enabled_t cleanupBeforePass;
  double searchFactorWhenWinning;
  double searchFactorWhenWinningThreshold;
};

static ContextConfig g_ctx;


struct GTPEngine {
  GTPEngine(const GTPEngine&) = delete;
  GTPEngine& operator=(const GTPEngine&) = delete;

  const string nnModelFile;
  const string humanModelFile;
  const bool assumeMultipleStartingBlackMovesAreHandicap;
  const int analysisPVLen;
  const bool preventEncore;

  const double dynamicPlayoutDoublingAdvantageCapPerOppLead;
  bool staticPDATakesPrecedence;
  double normalAvoidRepeatedPatternUtility;
  double handicapAvoidRepeatedPatternUtility;

  NNEvaluator* nnEval;
  NNEvaluator* humanEval;
  AsyncBot* bot;
  Rules currentRules; //Should always be the same as the rules in bot, if bot is not NULL.

  //Stores the params we want to be using during genmoves or analysis
  SearchParams genmoveParams;
  SearchParams analysisParams;
  bool isGenmoveParams;

  TimeControls bTimeControls;
  TimeControls wTimeControls;

  //This move history doesn't get cleared upon consecutive moves by the same side, and is used
  //for undo, whereas the one in search does.
  Board initialBoard;
  Player initialPla;
  vector<Move> moveHistory;

  vector<double> recentWinLossValues;
  double lastSearchFactor;
  double desiredDynamicPDAForWhite;

  double delayMoveScale;
  double delayMoveMax;

  Player perspective;

  Rand gtpRand;

  ClockTimer genmoveTimer;
  double genmoveTimeSum;
  std::atomic<int> genmoveExpectedId;

  //Positions during this game when genmove was called
  std::vector<Sgf::PositionSample> genmoveSamples;
  stringstream sstream;

  GTPEngine(
    const string& modelFile, const string& hModelFile,
    SearchParams initialGenmoveParams, SearchParams initialAnalysisParams,
    Rules initialRules,
    bool assumeMultiBlackHandicap, bool prevtEncore, 
    double dynamicPDACapPerOppLead, bool staticPDAPrecedence,
    double normAvoidRepeatedPatternUtility, double hcapAvoidRepeatedPatternUtility,
    double delayScale, double delayMax,
    Player persp, int pvLen
  )
    :nnModelFile(modelFile),
     humanModelFile(hModelFile),
     assumeMultipleStartingBlackMovesAreHandicap(assumeMultiBlackHandicap),
     analysisPVLen(pvLen),
     preventEncore(prevtEncore),
     dynamicPlayoutDoublingAdvantageCapPerOppLead(dynamicPDACapPerOppLead),
     staticPDATakesPrecedence(staticPDAPrecedence),
     normalAvoidRepeatedPatternUtility(normAvoidRepeatedPatternUtility),
     handicapAvoidRepeatedPatternUtility(hcapAvoidRepeatedPatternUtility),
     nnEval(NULL),
     humanEval(NULL),
     bot(NULL),
     currentRules(initialRules),
     genmoveParams(initialGenmoveParams),
     analysisParams(initialAnalysisParams),
     isGenmoveParams(true),
     bTimeControls(),
     wTimeControls(),
     initialBoard(),
     initialPla(P_BLACK),
     moveHistory(),
     recentWinLossValues(),
     lastSearchFactor(1.0),
     desiredDynamicPDAForWhite(0.0),
     delayMoveScale(delayScale),
     delayMoveMax(delayMax),
     perspective(persp),
     gtpRand(),
     genmoveTimer(),
     genmoveTimeSum(0.0),
     genmoveExpectedId(0),
     genmoveSamples()
  {
  }

  ~GTPEngine() {
    stopAndWait();
    delete bot;
    delete nnEval;
    delete humanEval;
  }

  void clearStream() {
    sstream.str("");
  }

  void stopAndWait() {
    // Invalidate any ongoing genmove
    int expectedSearchId = (genmoveExpectedId.load() + 1) & 0x3FFFFFFF;
    genmoveExpectedId.store(expectedSearchId);
    bot->stopAndWait();
  }

  Rules getCurrentRules() {
    return currentRules;
  }

  void clearStatsForNewGame() {
  }

  //Specify -1 for the sizes for a default
  void setOrResetBoardSize(ConfigParser& cfg, Logger& logger, Rand& seedRand, int boardXSize, int boardYSize) {
    bool wasDefault = false;
    if(boardXSize == -1 || boardYSize == -1) {
      boardXSize = Board::DEFAULT_LEN;
      boardYSize = Board::DEFAULT_LEN;
      wasDefault = true;
    }

    bool defaultRequireExactNNLen = true;
    int nnXLen = boardXSize;
    int nnYLen = boardYSize;

    if(cfg.contains("gtpForceMaxNNSize") && cfg.getBool("gtpForceMaxNNSize")) {
      defaultRequireExactNNLen = false;
      nnXLen = Board::MAX_LEN;
      nnYLen = Board::MAX_LEN;
    }

    //If the neural net is wrongly sized, we need to create or recreate it
    if(nnEval == NULL || !(nnXLen == nnEval->getNNXLen() && nnYLen == nnEval->getNNYLen())) {

      if(nnEval != NULL) {
        assert(bot != NULL);
        bot->stopAndWait();
        delete bot;
        delete nnEval;
        delete humanEval;
        bot = NULL;
        nnEval = NULL;
        humanEval = NULL;
        logger.write("Cleaned up old neural net and bot");
      }

      const int expectedConcurrentEvals = std::max(genmoveParams.numThreads, analysisParams.numThreads);
      const int defaultMaxBatchSize = std::max(8,((expectedConcurrentEvals+3)/4)*4);
      const bool disableFP16 = false;
      const string expectedSha256 = "";
      nnEval = Setup::initializeNNEvaluator(
        nnModelFile,nnModelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
        nnXLen,nnYLen,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_GTP
      );
      logger.write("Loaded neural net with nnXLen " + Global::intToString(nnEval->getNNXLen()) + " nnYLen " + Global::intToString(nnEval->getNNYLen()));
      if(humanModelFile != "") {
        humanEval = Setup::initializeNNEvaluator(
          humanModelFile,humanModelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
          nnXLen,nnYLen,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
          Setup::SETUP_FOR_GTP
        );
        logger.write("Loaded human SL net with nnXLen " + Global::intToString(humanEval->getNNXLen()) + " nnYLen " + Global::intToString(humanEval->getNNYLen()));
        if(!humanEval->requiresSGFMetadata()) {
          string warning;
          warning += "WARNING: Human model was not trained from SGF metadata to vary by rank! Did you pass the wrong model for -human-model?\n";
          logger.write(warning);
        }
      }

      {
        bool rulesWereSupported;
        nnEval->getSupportedRules(currentRules,rulesWereSupported);
        if(!rulesWereSupported) {
          throw StringError("Rules " + currentRules.toJsonStringNoKomi() + " from config file " + cfg.getFileName() + " are NOT supported by neural net");
        }
      }
    }

    //On default setup, also override board size to whatever the neural net was initialized with
    //So that if the net was initalized smaller, we don't fail with a big board
    if(wasDefault) {
      boardXSize = nnEval->getNNXLen();
      boardYSize = nnEval->getNNYLen();
    }

    //If the bot is wrongly sized, we need to create or recreate the bot
    if(bot == NULL || bot->getRootBoard().x_size != boardXSize || bot->getRootBoard().y_size != boardYSize) {
      if(bot != NULL) {
        assert(bot != NULL);
        bot->stopAndWait();
        delete bot;
        bot = NULL;
        logger.write("Cleaned up old bot");
      }

      logger.write("Initializing board with boardXSize " + Global::intToString(boardXSize) + " boardYSize " + Global::intToString(boardYSize));

      string searchRandSeed;
      if(cfg.contains("searchRandSeed"))
        searchRandSeed = cfg.getString("searchRandSeed");
      else
        searchRandSeed = Global::uint64ToString(seedRand.nextUInt64());

      bot = new AsyncBot(genmoveParams, nnEval, humanEval, &logger, searchRandSeed);
      bot->setCopyOfExternalPatternBonusTable(nullptr);

      Board board(boardXSize,boardYSize);
      Player pla = P_BLACK;
      BoardHistory hist(board,pla,currentRules,0);
      vector<Move> newMoveHistory;
      setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
      clearStatsForNewGame();
    }
  }

  void setPositionAndRules(Player pla, const Board& board, const BoardHistory& h, const Board& newInitialBoard, Player newInitialPla, const vector<Move> newMoveHistory) {
    BoardHistory hist(h);
    //Ensure we always have this value correct
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);

    currentRules = hist.rules;
    bot->setPosition(pla,board,hist);
    initialBoard = newInitialBoard;
    initialPla = newInitialPla;
    moveHistory = newMoveHistory;
    recentWinLossValues.clear();
    updateDynamicPDA();
  }

  void clearBoard() {
    assert(bot->getRootHist().rules == currentRules);
    int newXSize = bot->getRootBoard().x_size;
    int newYSize = bot->getRootBoard().y_size;
    Board board(newXSize,newYSize);
    Player pla = P_BLACK;
    BoardHistory hist(board,pla,currentRules,0);
    vector<Move> newMoveHistory;
    setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
    clearStatsForNewGame();
  }

  bool setPosition(const vector<Move>& initialStones) {
    assert(bot->getRootHist().rules == currentRules);
    int newXSize = bot->getRootBoard().x_size;
    int newYSize = bot->getRootBoard().y_size;
    Board board(newXSize,newYSize);
    bool suc = board.setStonesFailIfNoLibs(initialStones);
    if(!suc)
      return false;

    //Sanity check
    for(int i = 0; i<initialStones.size(); i++) {
      if(board.colors[initialStones[i].loc] != initialStones[i].pla) {
        assert(false);
        return false;
      }
    }
    Player pla = P_BLACK;
    BoardHistory hist(board,pla,currentRules,0);
    hist.setInitialTurnNumber(board.numStonesOnBoard()); //Heuristic to guess at what turn this is
    vector<Move> newMoveHistory;
    setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
    clearStatsForNewGame();
    return true;
  }

  void updateKomiIfNew(float newKomi) {
    bot->setKomiIfNew(newKomi);
    currentRules.komi = newKomi;
  }

  void updateDynamicPDA() {
    updateDynamicPDAHelper(
      bot->getRootBoard(),bot->getRootHist(),
      dynamicPlayoutDoublingAdvantageCapPerOppLead,
      recentWinLossValues,
      desiredDynamicPDAForWhite
    );
  }

  bool play(Loc loc, Player pla) {
    assert(bot->getRootHist().rules == currentRules);
    bool suc = bot->makeMove(loc,pla,preventEncore);
    if(suc)
      moveHistory.push_back(Move(loc,pla));
    return suc;
  }

  bool undo() {
    if(moveHistory.size() <= 0)
      return false;
    assert(bot->getRootHist().rules == currentRules);

    vector<Move> moveHistoryCopy = moveHistory;

    Board undoneBoard = initialBoard;
    BoardHistory undoneHist(undoneBoard,initialPla,currentRules,0);
    undoneHist.setInitialTurnNumber(bot->getRootHist().initialTurnNumber);
    vector<Move> emptyMoveHistory;
    setPositionAndRules(initialPla,undoneBoard,undoneHist,initialBoard,initialPla,emptyMoveHistory);

    for(int i = 0; i<moveHistoryCopy.size()-1; i++) {
      Loc moveLoc = moveHistoryCopy[i].loc;
      Player movePla = moveHistoryCopy[i].pla;
      bool suc = play(moveLoc,movePla);
      assert(suc);
      (void)suc; //Avoid warning when asserts are off
    }
    return true;
  }

  bool setRulesNotIncludingKomi(Rules newRules, string& error) {
    assert(nnEval != NULL);
    assert(bot->getRootHist().rules == currentRules);
    newRules.komi = currentRules.komi;

    bool rulesWereSupported;
    nnEval->getSupportedRules(newRules,rulesWereSupported);
    if(!rulesWereSupported) {
      error = "Rules " + newRules.toJsonStringNoKomi() + " are not supported by this neural net version";
      return false;
    }

    vector<Move> moveHistoryCopy = moveHistory;

    Board board = initialBoard;
    BoardHistory hist(board,initialPla,newRules,0);
    hist.setInitialTurnNumber(bot->getRootHist().initialTurnNumber);
    vector<Move> emptyMoveHistory;
    setPositionAndRules(initialPla,board,hist,initialBoard,initialPla,emptyMoveHistory);

    for(int i = 0; i<moveHistoryCopy.size(); i++) {
      Loc moveLoc = moveHistoryCopy[i].loc;
      Player movePla = moveHistoryCopy[i].pla;
      bool suc = play(moveLoc,movePla);

      //Because internally we use a highly tolerant test, we don't expect this to actually trigger
      //even if a rules change did make some earlier moves illegal. But this check simply futureproofs
      //things in case we ever do
      if(!suc) {
        error = "Could not make the rules change, some earlier moves in the game would now become illegal.";
        return false;
      }
    }
    return true;
  }

  void ponder() {
    bot->ponder(lastSearchFactor);
  }

  struct GenmoveArgs {
    double searchFactorWhenWinningThreshold;
    double searchFactorWhenWinning;
    enabled_t cleanupBeforePass;
    enabled_t friendlyPass;
  };

  struct AnalyzeArgs {
    bool analyzing = false;
    int minMoves = 0;
    int maxMoves = 10000000;
    double secondsPerReport = TimeControls::UNLIMITED_TIME_DEFAULT;
    vector<int> avoidMoveUntilByLocBlack;
    vector<int> avoidMoveUntilByLocWhite;
  };

  void filterZeroVisitMoves(const AnalyzeArgs& args, vector<AnalysisData> buf) {
    //Avoid printing moves that have 0 visits, unless we need them
    //These should already be sorted so that 0-visit moves only appear at the end.
    int keptMoves = 0;
    for(int i = 0; i<buf.size(); i++) {
      if(buf[i].childVisits > 0 || keptMoves < args.minMoves)
        buf[keptMoves++] = buf[i];
    }
    buf.resize(keptMoves);
  }

  std::function<void(const Search* search)> getAnalyzeCallback(Player pla, AnalyzeArgs args) {
    //Avoid capturing anything by reference except [this], since this will potentially be used
    //asynchronously and called after we return
    return [args,pla,this](const Search* search) {
      vector<AnalysisData> buf;
      bool duplicateForSymmetries = true;
      search->getAnalysisData(buf,args.minMoves,false,analysisPVLen,duplicateForSymmetries);
      filterZeroVisitMoves(args,buf);
      if(buf.size() > args.maxMoves)
        buf.resize(args.maxMoves);
      if(buf.size() <= 0)
        return;

      const Board board = search->getRootBoard();
      for(int i = 0; i<buf.size(); i++) {
        if(i > 0)
          sstream << ",";
        const AnalysisData& data = buf[i];
        double winrate = 0.5 * (1.0 + data.winLossValue);
        double lcb = PlayUtils::getHackedLCBForWinrate(search,data,pla);
        if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && pla == P_BLACK)) {
          winrate = 1.0-winrate;
          lcb = 1.0 - lcb;
        }
        sstream << Location::toString(data.move,board);
        sstream << " " << round(winrate * 10000.0) << " ";
        if(preventEncore && data.pvContainsPass())
          data.writePVUpToPhaseEnd(sstream,board,search->getRootHist(),search->getRootPla());
        else
          data.writePV(sstream,board);
      }
      sstream << endl;
    };
  }

  void genMove(
    Player pla,
    Logger& logger,
    const GenmoveArgs& gargs,
    const AnalyzeArgs& args,
    bool playChosenMove,
    bool& responseIsError,
    Loc& moveToPlay
  ) {
    bool onMoveWasCalled = false;
    Loc genmoveMoveLoc = Board::NULL_LOC;
    auto onMove = [&genmoveMoveLoc,&onMoveWasCalled,this](Loc moveLoc, int searchId, Search* search) noexcept {
      (void)searchId;
      (void)search;
      onMoveWasCalled = true;
      genmoveMoveLoc = moveLoc;
    };
    launchGenMove(pla,gargs,args,onMove);
    bot->waitForSearchToEnd();
    testAssert(onMoveWasCalled);
    string response;
    responseIsError = false;
    moveToPlay = Board::NULL_LOC;
    handleGenMoveResult(pla,bot->getSearchStopAndWait(),logger,gargs,args,genmoveMoveLoc,response, responseIsError,moveToPlay);
    if(!responseIsError && moveToPlay != Board::NULL_LOC && playChosenMove) {
      bool suc = bot->makeMove(moveToPlay,pla,preventEncore);
      if(suc)
        moveHistory.push_back(Move(moveToPlay,pla));
      assert(suc);
      (void)suc; //Avoid warning when asserts are off
    }
  }

  void genMoveCancellable(
    Player pla,
    Logger& logger,
    const GenmoveArgs& gargs,
    const AnalyzeArgs& args,
    std::function<void(const string&, bool)> printGTPResponse
  ) {
    // Make sure to capture things by value unless they're long-lived, since the callback needs to survive past the current scope.
    auto onMove = [pla,&logger,gargs,args,printGTPResponse,this](Loc moveLoc, int searchId, Search* search) {
      string response;
      bool responseIsError = false;
      Loc moveLocToPlay = Board::NULL_LOC;

      // Search invalidated before completion
      if(searchId != genmoveExpectedId.load()) {
        if(args.analyzing)
          response = "play cancelled";
        else
          response = "cancelled";
        printGTPResponse(response,responseIsError);
        return;
      }
      handleGenMoveResult(pla,search,logger,gargs,args,moveLoc,response,responseIsError,moveLocToPlay);
      printGTPResponse(response,responseIsError);
    };
    launchGenMove(pla,gargs,args,onMove);
  }

  void launchGenMove(
    Player pla,
    GenmoveArgs gargs,
    AnalyzeArgs args,
    std::function<void(Loc, int, Search*)> onMove
  ) {
    genmoveTimer.reset();

    nnEval->clearStats();
    if(humanEval != NULL)
      humanEval->clearStats();
    TimeControls tc = pla == P_BLACK ? bTimeControls : wTimeControls;

    if(!isGenmoveParams) {
      bot->setParams(genmoveParams);
      isGenmoveParams = true;
    }

    //Update dynamic PDA given whatever the most recent values are, if we're using dynamic
    updateDynamicPDA();

    SearchParams paramsToUse = genmoveParams;
    //Make sure we have the right parameters, in case someone updated params in the meantime.
    if(!staticPDATakesPrecedence) {
      double desiredDynamicPDA =
        (paramsToUse.playoutDoublingAdvantagePla == P_WHITE) ? desiredDynamicPDAForWhite :
        (paramsToUse.playoutDoublingAdvantagePla == P_BLACK) ? -desiredDynamicPDAForWhite :
        (paramsToUse.playoutDoublingAdvantagePla == C_EMPTY && pla == P_WHITE) ? desiredDynamicPDAForWhite :
        (paramsToUse.playoutDoublingAdvantagePla == C_EMPTY && pla == P_BLACK) ? -desiredDynamicPDAForWhite :
        (assert(false),0.0);

      paramsToUse.playoutDoublingAdvantage = desiredDynamicPDA;
    }

    {
      double avoidRepeatedPatternUtility = normalAvoidRepeatedPatternUtility;
      if(!args.analyzing) {
        double initialOppAdvantage = initialBlackAdvantage(bot->getRootHist()) * (pla == P_WHITE ? 1 : -1);
        if(initialOppAdvantage > getPointsThresholdForHandicapGame(getBoardSizeScaling(bot->getRootBoard())))
          avoidRepeatedPatternUtility = handicapAvoidRepeatedPatternUtility;
      }
      paramsToUse.avoidRepeatedPatternUtility = avoidRepeatedPatternUtility;
    }

    if(paramsToUse != bot->getParams())
      bot->setParams(paramsToUse);


    //Play faster when winning
    double searchFactor = PlayUtils::getSearchFactor(gargs.searchFactorWhenWinningThreshold,gargs.searchFactorWhenWinning,paramsToUse,recentWinLossValues,pla);
    lastSearchFactor = searchFactor;

    bot->setAvoidMoveUntilByLoc(args.avoidMoveUntilByLocBlack,args.avoidMoveUntilByLocWhite);

    //So that we can tell by the end of the search whether we still care for the result.
    int expectedSearchId = (genmoveExpectedId.load() + 1) & 0x3FFFFFFF;
    genmoveExpectedId.store(expectedSearchId);

    if(args.analyzing) {
      std::function<void(const Search* search)> callback = getAnalyzeCallback(pla,args);
      bot->setAlwaysIncludeOwnerMap(false);

      //Make sure callback happens at least once
      auto onMoveWrapped = [onMove,callback](Loc moveLoc, int searchId, Search* search) {
        callback(search);
        onMove(moveLoc,searchId,search);
      };
      bot->genMoveAsyncAnalyze(pla, expectedSearchId, tc, searchFactor, onMoveWrapped, args.secondsPerReport, args.secondsPerReport, callback);
    }
    else {
      bot->genMoveAsync(pla,expectedSearchId,tc,searchFactor,onMove);
    }
  }

  void handleGenMoveResult(
    Player pla,
    Search* searchBot,
    Logger& logger,
    const GenmoveArgs& gargs,
    const AnalyzeArgs& args,
    Loc moveLoc,
    string& response, bool& responseIsError,
    Loc& moveLocToPlay
  ) {
    response = "";
    responseIsError = false;
    moveLocToPlay = Board::NULL_LOC;

    const Search* search = searchBot;

    bool isLegal = search->isLegalStrict(moveLoc,pla);
    if(moveLoc == Board::NULL_LOC || !isLegal) {
      responseIsError = true;
      response = "genmove returned null location or illegal move";
      ostringstream sout;
      sout << "genmove null location or illegal move!?!" << "\n";
      sout << search->getRootBoard() << "\n";
      sout << "Pla: " << PlayerIO::playerToString(pla) << "\n";
      sout << "MoveLoc: " << Location::toString(moveLoc,search->getRootBoard()) << "\n";
      logger.write(sout.str());
      genmoveTimeSum += genmoveTimer.getSeconds();
      return;
    }

    SearchNode* rootNode = search->rootNode;
    if(rootNode != NULL && delayMoveScale > 0.0 && delayMoveMax > 0.0) {
      int pos = search->getPos(moveLoc);
      const NNOutput* nnOutput = rootNode->getHumanOutput();
      nnOutput = nnOutput != NULL ? nnOutput : rootNode->getNNOutput();
      const float* policyProbs = nnOutput != NULL ? nnOutput->getPolicyProbsMaybeNoised() : NULL;
      if(policyProbs != NULL) {
        double prob = std::max(0.0,(double)policyProbs[pos]);
        double meanWait = 0.5 * delayMoveScale / (prob + 0.10);
        double waitTime = gtpRand.nextGamma(2.0) * meanWait / 2.0;
        waitTime = std::min(waitTime,delayMoveMax);
        waitTime = std::max(waitTime,0.0001);
        std::this_thread::sleep_for(std::chrono::duration<double>(waitTime));
      }
    }

    ReportedSearchValues values;
    double winLossValue;
    double lead;
    {
      values = search->getRootValuesRequireSuccess();
      winLossValue = values.winLossValue;
      lead = values.lead;
    }

    //Record data for resignation or adjusting handicap behavior ------------------------
    recentWinLossValues.push_back(winLossValue);

    //Decide whether we should resign---------------------
    //Snapshot the time NOW - all meaningful play-related computation time is done, the rest is just
    //output of various things.
    double timeTaken = genmoveTimer.getSeconds();
    genmoveTimeSum += timeTaken;

    const SearchParams& params = search->searchParams;


    //Hacks--------------------------------------------------
    //At least one of these hacks will use the bot to search stuff and clears its tree, so we apply them AFTER
    //all relevant logging and stuff.

    //Implement friendly pass - in area scoring rules other than tromp-taylor, maybe pass once there are no points
    //left to gain.
    int64_t numVisitsForFriendlyPass = 8 + std::min((int64_t)1000, std::min(params.maxVisits, params.maxPlayouts) / 10);
    moveLoc = PlayUtils::maybeFriendlyPass(gargs.cleanupBeforePass, gargs.friendlyPass, pla, moveLoc, searchBot, numVisitsForFriendlyPass);

    //Implement cleanupBeforePass hack - if the bot wants to pass, instead cleanup if there is something to clean
    //and we are in a ruleset where this is necessary or the user has configured it.
    moveLoc = PlayUtils::maybeCleanupBeforePass(gargs.cleanupBeforePass, gargs.friendlyPass, pla, moveLoc, bot);

    //Actual reporting of chosen move---------------------

    // I assume resigned is never called in JNI.
    moveLocToPlay = moveLoc;

    if(args.analyzing) {
      response = "play " + response;
    }

    return;
  }

  void clearCache() {
    bot->clearSearch();
    nnEval->clearCache();
    if(humanEval != NULL)
      humanEval->clearCache();
  }

  void placeFixedHandicap(int n, string& response, bool& responseIsError) {
    int xSize = bot->getRootBoard().x_size;
    int ySize = bot->getRootBoard().y_size;
    Board board(xSize,ySize);
    try {
      PlayUtils::placeFixedHandicap(board,n);
    }
    catch(const StringError& e) {
      responseIsError = true;
      response = string(e.what()) + ", try place_free_handicap";
      return;
    }
    assert(bot->getRootHist().rules == currentRules);

    Player pla = P_BLACK;
    BoardHistory hist(board,pla,currentRules,0);

    //Also switch the initial player, expecting white should be next.
    hist.clear(board,P_WHITE,currentRules,0);
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);
    hist.setInitialTurnNumber(board.numStonesOnBoard()); //Should give more accurate temperaure and time control behavior
    pla = P_WHITE;

    response = "";
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] != C_EMPTY) {
          response += " " + Location::toString(loc,board);
        }
      }
    }
    response = Global::trim(response);
    (void)responseIsError;

    vector<Move> newMoveHistory;
    setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
    clearStatsForNewGame();
  }

  void placeFreeHandicap(int n, string& response, bool& responseIsError, Rand& rand) {
    stopAndWait();

    //If asked to place more, we just go ahead and only place up to 30, or a quarter of the board
    int xSize = bot->getRootBoard().x_size;
    int ySize = bot->getRootBoard().y_size;
    int maxHandicap = xSize*ySize / 4;
    if(maxHandicap > 30)
      maxHandicap = 30;
    if(n > maxHandicap)
      n = maxHandicap;

    assert(bot->getRootHist().rules == currentRules);

    Board board(xSize,ySize);
    Player pla = P_BLACK;
    BoardHistory hist(board,pla,currentRules,0);
    double extraBlackTemperature = 0.25;
    PlayUtils::playExtraBlack(bot->getSearchStopAndWait(), n, board, hist, extraBlackTemperature, rand);
    //Also switch the initial player, expecting white should be next.
    hist.clear(board,P_WHITE,currentRules,0);
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);
    hist.setInitialTurnNumber(board.numStonesOnBoard()); //Should give more accurate temperaure and time control behavior
    pla = P_WHITE;

    response = "";
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] != C_EMPTY) {
          response += " " + Location::toString(loc,board);
        }
      }
    }
    response = Global::trim(response);
    (void)responseIsError;

    vector<Move> newMoveHistory;
    setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
    clearStatsForNewGame();
  }

  void analyze(Player pla, AnalyzeArgs args) {
    clearStream();
    assert(args.analyzing);
    if(isGenmoveParams) {
      bot->setParams(analysisParams);
      isGenmoveParams = false;
    }

    std::function<void(const Search* search)> callback = getAnalyzeCallback(pla,args);
    bot->setAvoidMoveUntilByLoc(args.avoidMoveUntilByLocBlack,args.avoidMoveUntilByLocWhite);
    bot->setAlwaysIncludeOwnerMap(false);

    double searchFactor = 1e40; //go basically forever
    bot->analyzeAsync(pla, searchFactor, args.secondsPerReport, args.secondsPerReport, callback);
  }

  void computeAnticipatedWinnerAndScore(Player& winner, double& finalWhiteMinusBlackScore) {
    stopAndWait();

    //No playoutDoublingAdvantage to avoid bias
    //Also never assume the game will end abruptly due to pass
    {
      SearchParams tmpParams = genmoveParams;
      tmpParams.playoutDoublingAdvantage = 0.0;
      tmpParams.conservativePass = true;
      tmpParams.humanSLChosenMoveProp = 0.0;
      tmpParams.humanSLRootExploreProbWeightful = 0.0;
      tmpParams.humanSLRootExploreProbWeightless = 0.0;
      tmpParams.humanSLPlaExploreProbWeightful = 0.0;
      tmpParams.humanSLPlaExploreProbWeightless = 0.0;
      tmpParams.humanSLOppExploreProbWeightful = 0.0;
      tmpParams.humanSLOppExploreProbWeightless = 0.0;
      tmpParams.antiMirror = false;
      tmpParams.avoidRepeatedPatternUtility = 0;
      bot->setParams(tmpParams);
    }

    //Make absolutely sure we can restore the bot's old state
    const Player oldPla = bot->getRootPla();
    const Board oldBoard = bot->getRootBoard();
    const BoardHistory oldHist = bot->getRootHist();

    Board board = bot->getRootBoard();
    BoardHistory hist = bot->getRootHist();
    Player pla = bot->getRootPla();

    //Tromp-taylorish scoring, or finished territory game scoring (including noresult)
    if(hist.isGameFinished && (
         (hist.rules.scoringRule == Rules::SCORING_AREA && !hist.rules.friendlyPassOk) ||
         (hist.rules.scoringRule == Rules::SCORING_TERRITORY)
       )
    ) {
      //For GTP purposes, we treat noResult as a draw since there is no provision for anything else.
      winner = hist.winner;
      finalWhiteMinusBlackScore = hist.finalWhiteMinusBlackScore;
    }
    //Human-friendly score or incomplete game score estimation
    else {
      int64_t numVisits = std::max(50, genmoveParams.numThreads * 10);
      //Try computing the lead for white
      double lead = PlayUtils::computeLead(bot->getSearchStopAndWait(),NULL,board,hist,pla,numVisits,OtherGameProperties());

      //Round lead to nearest integer or half-integer
      if(hist.rules.gameResultWillBeInteger())
        lead = round(lead);
      else
        lead = round(lead+0.5)-0.5;

      finalWhiteMinusBlackScore = lead;
      winner = lead > 0 ? P_WHITE : lead < 0 ? P_BLACK : C_EMPTY;
    }

    //Restore
    bot->setPosition(oldPla,oldBoard,oldHist);
    bot->setParams(genmoveParams);
    isGenmoveParams = true;
  }

  vector<bool> computeAnticipatedStatuses() {
    stopAndWait();

    //No playoutDoublingAdvantage to avoid bias
    //Also never assume the game will end abruptly due to pass
    {
      SearchParams tmpParams = genmoveParams;
      tmpParams.playoutDoublingAdvantage = 0.0;
      tmpParams.conservativePass = true;
      tmpParams.humanSLChosenMoveProp = 0.0;
      tmpParams.humanSLRootExploreProbWeightful = 0.0;
      tmpParams.humanSLRootExploreProbWeightless = 0.0;
      tmpParams.humanSLPlaExploreProbWeightful = 0.0;
      tmpParams.humanSLPlaExploreProbWeightless = 0.0;
      tmpParams.humanSLOppExploreProbWeightful = 0.0;
      tmpParams.humanSLOppExploreProbWeightless = 0.0;
      tmpParams.antiMirror = false;
      tmpParams.avoidRepeatedPatternUtility = 0;
      bot->setParams(tmpParams);
    }

    //Make absolutely sure we can restore the bot's old state
    const Player oldPla = bot->getRootPla();
    const Board oldBoard = bot->getRootBoard();
    const BoardHistory oldHist = bot->getRootHist();

    Board board = bot->getRootBoard();
    BoardHistory hist = bot->getRootHist();
    Player pla = bot->getRootPla();

    int64_t numVisits = std::max(100, genmoveParams.numThreads * 20);
    vector<bool> isAlive;
    //Tromp-taylorish statuses, or finished territory game statuses (including noresult)
    if(hist.isGameFinished && (
         (hist.rules.scoringRule == Rules::SCORING_AREA && !hist.rules.friendlyPassOk) ||
         (hist.rules.scoringRule == Rules::SCORING_TERRITORY)
       )
    )
      isAlive = PlayUtils::computeAnticipatedStatusesSimple(board,hist);
    //Human-friendly statuses or incomplete game status estimation
    else {
      vector<double> ownershipsBuf;
      isAlive = PlayUtils::computeAnticipatedStatusesWithOwnership(bot->getSearchStopAndWait(),board,hist,pla,numVisits,ownershipsBuf);
    }

    //Restore
    bot->setPosition(oldPla,oldBoard,oldHist);
    bot->setParams(genmoveParams);
    isGenmoveParams = true;

    return isAlive;
  }

  string rawNNBrief(std::vector<Loc> branch, int whichSymmetry) {
    if(nnEval == NULL)
      return "";
    ostringstream out;

    Player pla = bot->getRootPla();
    Board board = bot->getRootBoard();
    BoardHistory hist = bot->getRootHist();

    Player prevPla = pla;
    Board prevBoard = board;
    BoardHistory prevHist = hist;
    Loc prevLoc = Board::NULL_LOC;

    for(Loc loc: branch) {
      prevPla = pla;
      prevBoard = board;
      prevHist = hist;
      prevLoc = loc;
      bool suc = hist.makeBoardMoveTolerant(board, loc, pla, false);
      if(!suc)
        return "illegal move sequence";
      pla = getOpp(pla);
    }

    string policyStr = "Policy: ";
    string wlStr = "White winloss: ";
    string leadStr = "White lead: ";

    for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
      if(whichSymmetry == NNInputs::SYMMETRY_ALL || whichSymmetry == symmetry) {
        {
          MiscNNInputParams nnInputParams;
          nnInputParams.playoutDoublingAdvantage =
            (analysisParams.playoutDoublingAdvantagePla == C_EMPTY || analysisParams.playoutDoublingAdvantagePla == pla) ?
            analysisParams.playoutDoublingAdvantage : -analysisParams.playoutDoublingAdvantage;
          nnInputParams.symmetry = symmetry;

          NNResultBuf buf;
          bool skipCache = true;
          bool includeOwnerMap = false;
          nnEval->evaluate(board,hist,pla,&analysisParams.humanSLProfile,nnInputParams,buf,skipCache,includeOwnerMap);

          NNOutput* nnOutput = buf.result.get();
          wlStr += Global::strprintf("%.2fc ", 100.0 * (nnOutput->whiteWinProb - nnOutput->whiteLossProb));
          leadStr += Global::strprintf("%.2f ", nnOutput->whiteLead);
        }
        if(prevLoc != Board::NULL_LOC) {
          MiscNNInputParams nnInputParams;
          nnInputParams.playoutDoublingAdvantage =
            (analysisParams.playoutDoublingAdvantagePla == C_EMPTY || analysisParams.playoutDoublingAdvantagePla == prevPla) ?
            analysisParams.playoutDoublingAdvantage : -analysisParams.playoutDoublingAdvantage;
          nnInputParams.symmetry = symmetry;

          NNResultBuf buf;
          bool skipCache = true;
          bool includeOwnerMap = false;
          nnEval->evaluate(prevBoard,prevHist,prevPla,&analysisParams.humanSLProfile,nnInputParams,buf,skipCache,includeOwnerMap);

          NNOutput* nnOutput = buf.result.get();
          int pos = NNPos::locToPos(prevLoc,board.x_size,nnOutput->nnXLen,nnOutput->nnYLen);
          policyStr += Global::strprintf("%.2f%% ", 100.0 * (nnOutput->policyProbs[pos]));
        }
      }
    }
    return Global::trim(policyStr + "\n" + wlStr + "\n" + leadStr);
  }

  string rawNN(int whichSymmetry, double policyOptimism, bool useHumanModel) {
    NNEvaluator* nnEvalToUse = useHumanModel ? humanEval : nnEval;
    if(nnEvalToUse == NULL)
      return "";
    ostringstream out;

    for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
      if(whichSymmetry == NNInputs::SYMMETRY_ALL || whichSymmetry == symmetry) {
        Board board = bot->getRootBoard();
        BoardHistory hist = bot->getRootHist();
        Player nextPla = bot->getRootPla();

        MiscNNInputParams nnInputParams;
        nnInputParams.playoutDoublingAdvantage =
          (analysisParams.playoutDoublingAdvantagePla == C_EMPTY || analysisParams.playoutDoublingAdvantagePla == nextPla) ?
          analysisParams.playoutDoublingAdvantage : -analysisParams.playoutDoublingAdvantage;
        nnInputParams.symmetry = symmetry;
        nnInputParams.policyOptimism = policyOptimism;
        NNResultBuf buf;
        bool skipCache = true;
        bool includeOwnerMap = true;
        nnEvalToUse->evaluate(board,hist,nextPla,&analysisParams.humanSLProfile,nnInputParams,buf,skipCache,includeOwnerMap);

        NNOutput* nnOutput = buf.result.get();
        out << "symmetry " << symmetry << endl;
        out << "whiteWin " << Global::strprintf("%.6f",nnOutput->whiteWinProb) << endl;
        out << "whiteLoss " << Global::strprintf("%.6f",nnOutput->whiteLossProb) << endl;
        out << "noResult " << Global::strprintf("%.6f",nnOutput->whiteNoResultProb) << endl;
        if(useHumanModel) {
          out << "whiteScore " << Global::strprintf("%.3f",nnOutput->whiteScoreMean) << endl;
          out << "whiteScoreSq " << Global::strprintf("%.3f",nnOutput->whiteScoreMeanSq) << endl;
        }
        else {
          out << "whiteLead " << Global::strprintf("%.3f",nnOutput->whiteLead) << endl;
          out << "whiteScoreSelfplay " << Global::strprintf("%.3f",nnOutput->whiteScoreMean) << endl;
          out << "whiteScoreSelfplaySq " << Global::strprintf("%.3f",nnOutput->whiteScoreMeanSq) << endl;
          out << "varTimeLeft " << Global::strprintf("%.3f",nnOutput->varTimeLeft) << endl;
        }
        out << "shorttermWinlossError " << Global::strprintf("%.3f",nnOutput->shorttermWinlossError) << endl;
        out << "shorttermScoreError " << Global::strprintf("%.3f",nnOutput->shorttermScoreError) << endl;

        out << "policy" << endl;
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            int pos = NNPos::xyToPos(x,y,nnOutput->nnXLen);
            float prob = nnOutput->policyProbs[pos];
            if(prob < 0)
              out << "    NAN ";
            else
              out << Global::strprintf("%8.6f ", prob);
          }
          out << endl;
        }
        out << "policyPass ";
        {
          int pos = NNPos::locToPos(Board::PASS_LOC,board.x_size,nnOutput->nnXLen,nnOutput->nnYLen);
          float prob = nnOutput->policyProbs[pos];
          if(prob < 0)
            out << "    NAN "; // Probably shouldn't ever happen for pass unles the rules change, but we handle it anyways
          else
            out << Global::strprintf("%8.6f ", prob);
          out << endl;
        }

        out << "whiteOwnership" << endl;
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            int pos = NNPos::xyToPos(x,y,nnOutput->nnXLen);
            float whiteOwn = nnOutput->whiteOwnerMap[pos];
            out << Global::strprintf("%9.7f ", whiteOwn);
          }
          out << endl;
        }
        out << endl;
      }
    }

    return Global::trim(out.str());
  }

  const SearchParams& getGenmoveParams() {
    return genmoveParams;
  }

  void setGenmoveParamsIfChanged(const SearchParams& p) {
    if(genmoveParams != p) {
      genmoveParams = p;
      if(isGenmoveParams)
        bot->setParams(genmoveParams);
    }
  }

  const SearchParams& getAnalysisParams() {
    return analysisParams;
  }

  void setAnalysisParamsIfChanged(const SearchParams& p) {
    if(analysisParams != p) {
      analysisParams = p;
      if(!isGenmoveParams)
        bot->setParams(analysisParams);
    }
  }
};

/*
  Our own implementation.
*/

static std::unique_ptr<GTPEngine> g_engine;
static std::unique_ptr<Logger> g_logger;
static ConfigParser g_cfg;
static Rand g_seedRand;

// Similar to MainCmds::gtp
static void JNISetup( int threadNum, const std::string& cfgPath, const std::string& modelPath )
{
  Board::initHash();
  ScoreValue::initTables();
  Rand& seedRand = g_seedRand;

  ConfigParser& cfg = g_cfg;

  cfg.initialize(cfgPath);
  cfg.overrideKey("numSearchThreads", std::to_string(threadNum) );

  g_logger.reset( new Logger(&cfg) );
  Logger& logger = *g_logger;

  Rules initialRules = Setup::loadSingleRules(cfg, false);

  auto loadParams = [](ConfigParser& config, SearchParams& genmoveOut, SearchParams& analysisOut) {
    SearchParams params = Setup::loadSingleParams(config,Setup::SETUP_FOR_GTP);
    //Set a default for conservativePass that differs from matches or selfplay
    if(!config.contains("conservativePass"))
      params.conservativePass = true;
    if(!config.contains("fillDameBeforePass"))
      params.fillDameBeforePass = true;

    const double analysisWideRootNoise =
      config.contains("analysisWideRootNoise") ? config.getDouble("analysisWideRootNoise",0.0,5.0) : Setup::DEFAULT_ANALYSIS_WIDE_ROOT_NOISE;
    const bool analysisIgnorePreRootHistory =
      config.contains("analysisIgnorePreRootHistory") ? config.getBool("analysisIgnorePreRootHistory") : Setup::DEFAULT_ANALYSIS_IGNORE_PRE_ROOT_HISTORY;
    const bool genmoveAntiMirror =
      config.contains("genmoveAntiMirror") ? config.getBool("genmoveAntiMirror") : config.contains("antiMirror") ? config.getBool("antiMirror") : true;

    genmoveOut = params;
    analysisOut = params;

    genmoveOut.antiMirror = genmoveAntiMirror;
    analysisOut.wideRootNoise = analysisWideRootNoise;
    analysisOut.ignorePreRootHistory = analysisIgnorePreRootHistory;
  };

  SearchParams initialGenmoveParams;
  SearchParams initialAnalysisParams;
  loadParams(cfg,initialGenmoveParams,initialAnalysisParams);

  g_ctx.cleanupBeforePass = cfg.contains("cleanupBeforePass") ? cfg.getEnabled("cleanupBeforePass") : enabled_t::Auto;
  g_ctx.friendlyPass = cfg.contains("friendlyPass") ? cfg.getEnabled("friendlyPass") : enabled_t::Auto;
  if(g_ctx.cleanupBeforePass == enabled_t::True && g_ctx.friendlyPass == enabled_t::True)
    throw StringError("Cannot specify both cleanupBeforePass = true and friendlyPass = true at the same time");

  Setup::initializeSession(cfg);

  g_ctx.searchFactorWhenWinning = cfg.contains("searchFactorWhenWinning") ? cfg.getDouble("searchFactorWhenWinning",0.01,1.0) : 1.0;
  g_ctx.searchFactorWhenWinningThreshold = cfg.contains("searchFactorWhenWinningThreshold") ? cfg.getDouble("searchFactorWhenWinningThreshold",0.0,1.0) : 1.0;

  const int analysisPVLen = cfg.contains("analysisPVLen") ? cfg.getInt("analysisPVLen",1,1000) : 13;
  const bool assumeMultipleStartingBlackMovesAreHandicap =
    cfg.contains("assumeMultipleStartingBlackMovesAreHandicap") ? cfg.getBool("assumeMultipleStartingBlackMovesAreHandicap") : true;
  const bool preventEncore = cfg.contains("preventCleanupPhase") ? cfg.getBool("preventCleanupPhase") : true;
  const double dynamicPlayoutDoublingAdvantageCapPerOppLead =
    cfg.contains("dynamicPlayoutDoublingAdvantageCapPerOppLead") ? cfg.getDouble("dynamicPlayoutDoublingAdvantageCapPerOppLead",0.0,0.5) : 0.045;
  bool staticPDATakesPrecedence = cfg.contains("playoutDoublingAdvantage") && !cfg.contains("dynamicPlayoutDoublingAdvantageCapPerOppLead");
  const double normalAvoidRepeatedPatternUtility = initialGenmoveParams.avoidRepeatedPatternUtility;
  const double handicapAvoidRepeatedPatternUtility = cfg.contains("avoidRepeatedPatternUtility") ?
    initialGenmoveParams.avoidRepeatedPatternUtility : 0.005;
  const double initialDelayMoveScale = cfg.contains("delayMoveScale") ? cfg.getDouble("delayMoveScale",0.0,10000.0) : 0.0;
  const double initialDelayMoveMax = cfg.contains("delayMoveMax") ? cfg.getDouble("delayMoveMax",0.0,1000000.0) : 1000000.0;

  int defaultBoardXSize = -1;
  int defaultBoardYSize = -1;
  Setup::loadDefaultBoardXYSize(cfg,logger,defaultBoardXSize,defaultBoardYSize);


  Player perspective = Setup::parseReportAnalysisWinrates(cfg,C_EMPTY);

  g_engine.reset(new GTPEngine(
    modelPath,"",
    initialGenmoveParams,initialAnalysisParams,
    initialRules,
    assumeMultipleStartingBlackMovesAreHandicap,preventEncore,    dynamicPlayoutDoublingAdvantageCapPerOppLead,
    staticPDATakesPrecedence,
    normalAvoidRepeatedPatternUtility, handicapAvoidRepeatedPatternUtility,
    initialDelayMoveScale,initialDelayMoveMax,
    perspective,analysisPVLen
  ));

  GTPEngine *engine = g_engine.get();
  engine->setOrResetBoardSize(cfg,logger,seedRand,defaultBoardXSize,defaultBoardYSize);

  //If nobody specified any time limit in any way, then assume a relatively fast time control
  if(!cfg.contains("maxPlayouts") && !cfg.contains("maxVisits") && !cfg.contains("maxTime")) {
    double mainTime = 1.0;
    double byoYomiTime = 5.0;
    int byoYomiPeriods = 5;
    TimeControls tc = TimeControls::canadianOrByoYomiTime(mainTime,byoYomiTime,byoYomiPeriods,1);
    engine->bTimeControls = tc;
    engine->wTimeControls = tc;
  }

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);
  Setup::maybeWarnHumanSLParams(initialGenmoveParams,engine->nnEval,engine->humanEval,cerr,&logger);
}

extern "C"
{

void
Java_io_github_karino2_paoogo_goengine_katago_KataGoNative_initNative (
	JNIEnv*	env,
	jclass clasz,
  jint threadNum,
  jstring cfgPath,
  jstring modelPath
	)
{
  if (g_engine.get() != nullptr)
    return;

  const char* cfgCStr = env->GetStringUTFChars(cfgPath, nullptr);
  std::string cfgPathS(cfgCStr);
  env->ReleaseStringUTFChars(cfgPath, cfgCStr);

  const char* modelCStr = env->GetStringUTFChars(modelPath, nullptr);
  std::string modelPathS(modelCStr);
  env->ReleaseStringUTFChars(modelPath, modelCStr);

  
  __android_log_print(ANDROID_LOG_VERBOSE, "PaooGo",  "cfg=%s, model=%s", cfgPathS.c_str(), modelPathS.c_str());

  JNISetup( threadNum, cfgPathS, modelPathS );
}

void
Java_io_github_karino2_paoogo_goengine_katago_KataGoNative_setKomi (
	JNIEnv*	env,
	jclass clasz,
	jfloat komi
	)
{
  g_engine->updateKomiIfNew(komi);
}


void
Java_io_github_karino2_paoogo_goengine_katago_KataGoNative_clearBoard (
	JNIEnv*	env,
	jclass clasz
	)
{
  g_engine->clearBoard();
}

static Player IsBlackToColor( jboolean isBlack ) {
  if (isBlack) {
    return P_BLACK;
  } else{
    return P_WHITE;
  }

}

/*
  x, y is start from upperleft, 0 origin.

  A3 means
  x = 0
  y = BOARD_SIZE-3
*/
jboolean
Java_io_github_karino2_paoogo_goengine_katago_KataGoNative_doMove (
	JNIEnv*	env,
	jclass clasz,
  jint x, 
  jint y,
  jboolean isBlack
	)
{
  Player pla = IsBlackToColor(isBlack);
  Loc loc;

  loc = Location::getLoc(x, y, g_engine->bot->getRootBoard().x_size);

  return g_engine->play(loc, pla);
}

void
Java_io_github_karino2_paoogo_goengine_katago_KataGoNative_doPass (
	JNIEnv*	env,
	jclass clasz,
  jboolean isBlack
	)
{
  Player pla = IsBlackToColor(isBlack);
  Loc loc = Board::PASS_LOC;
  g_engine->play(loc, pla);
}

/*
  x | (y<<16),
  pass is -1.
*/
jint
Java_io_github_karino2_paoogo_goengine_katago_KataGoNative_genMoveInternal (
	JNIEnv*	env,
	jclass clasz,
  jboolean isBlack
	)
{
  Player pla = IsBlackToColor(isBlack);

  GTPEngine::GenmoveArgs gargs;
  gargs.searchFactorWhenWinningThreshold = g_ctx.searchFactorWhenWinningThreshold;
  gargs.searchFactorWhenWinning = g_ctx.searchFactorWhenWinning;
  gargs.cleanupBeforePass = g_ctx.cleanupBeforePass;
  gargs.friendlyPass = g_ctx.friendlyPass;

  bool isError;
  Loc moveLoc;
  g_engine->genMove(pla, *g_logger, gargs, GTPEngine::AnalyzeArgs(), true, isError, moveLoc );
  if (isError || moveLoc == Board::PASS_LOC) {
    // treat error as PASS.
    return -1;
  }

  int x_size = g_engine->bot->getRootBoard().x_size;
  int x = Location::getX( moveLoc, x_size );
  int y = Location::getY( moveLoc, x_size );
  return x | (y<<16);
}


void
Java_io_github_karino2_paoogo_goengine_katago_KataGoNative_setBoardSize (
	JNIEnv*	env,
	jclass clasz,
	jint boardSize
	)
{
  g_engine->setOrResetBoardSize(g_cfg,*g_logger,g_seedRand,boardSize,boardSize);
}

void
Java_io_github_karino2_paoogo_goengine_katago_KataGoNative_setGenmoveProfile (
	JNIEnv*	env,
	jclass clasz,
	jstring profile
	)
{
  const char* profileCStr = env->GetStringUTFChars(profile, nullptr);
  std::string profileS(profileCStr);
  env->ReleaseStringUTFChars(profile, profileCStr);
  SearchParams genmoveParams = g_engine->getGenmoveParams();
  genmoveParams.humanSLProfile = SGFMetadata::getProfile(profileS);
  g_engine->setGenmoveParamsIfChanged(genmoveParams);
}

// 固定時間で同期的に振る舞う関数。文字列で結果を返す。
// lz-analyzeを元にした文字列。
// [KataGo/docs/GTP_Extensions.md at master · lightvector/KataGo](https://github.com/lightvector/KataGo/blob/master/docs/GTP_Extensions.md)
// winrate: [0, 10000]
// 先頭が候補手とwinrate、そのあとにpvが続く。空白区切り。
// pvが終わったらカンマ区切りで次のinfo。
// 例としては以下のような出力。
// F4 5506 F4 E2 G4 E7 F7 H3,D6 5546 D6 D7 C7 E6 D5 E7 E5 C8 B7 B8 B6,D7 5341 D7 D6 E6 C7 D8 B4,G5 5099 G5 D2 D7 C7 C3,C5 5092 C5 D7 F4 H5,E3 4829 E3 H5 H6 G5 G6 D7,E7 4955 E7 D7 F4 E2,E5 4706 E5 B4 H4,B5 4847 B5 E5,G4 4320 G4,F3 4450 F3,E2 4238 E2,C7 4212 C7,H5 4164 H5
jstring
Java_io_github_karino2_paoogo_goengine_katago_KataGoNative_analyze (
	JNIEnv*	env,
	jclass clasz,
  jint msec,
  jboolean isBlack
	)
{
  Player pla = IsBlackToColor(isBlack);
  GTPEngine::AnalyzeArgs args;
  args.analyzing = true;
  args.secondsPerReport = (((double)msec)/1000.0) - 0.1;
  args.minMoves = 0;
  args.maxMoves = 10000000;
  args.avoidMoveUntilByLocBlack = vector<int>{};
  args.avoidMoveUntilByLocWhite = vector<int>{};

  g_engine->analyze(pla, args);

  auto start_time = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start_time < std::chrono::milliseconds(msec)) {
     std::this_thread::yield();
  }

  return env->NewStringUTF(g_engine->sstream.str().c_str());
}


}
