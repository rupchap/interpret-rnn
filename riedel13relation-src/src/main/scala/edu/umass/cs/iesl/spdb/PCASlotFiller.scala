package edu.umass.cs.iesl.spdb

import cc.factorie.db.mongo.MongoCubbieImplicits._
import java.io.{File, PrintStream}
import collection.mutable.{HashSet, HashMap, ArrayBuffer}

/**
 * The PCASlotFiller creates a matrix of candidate tuples vs surface patterns, slots and freebase relations.
 * Then performs PCA (or variants of it) and predicts the unseen cells that have query/test entities as argument one.
 *
 * The PCASlotFiller expects a prepared Database.CorefDB and Database.KBDB. This means that mentions, entities
 * (including TAC query entities), TAC training relations, freebase relations, as well as surface patterns (relation
 * mentions) need to be in place already.
 *
 * @author sriedel
 */
object PCASlotFiller extends HasLogger {

  /**
   * This method loads the relation mentions corresponding to a set of documents, and adds these to the matrix.
   * @param spdb the matrix.
   * @param docs the docs from where to load the relation mentions
   * @param hidden should the cells be observed or hidden.
   * @param onlyAddToExistingTuples should we only add a cell if the tuple has already at least one cell.
   * @param minCellsPerRelation minimum number of cells a relation needs to have to be incorporated into the matrix.
   * @param minCellsPerTuple minimum number of cells a tuple needs to have to become a row in the matrix.
   */
  def addRelationMentionsTo(spdb: SelfPredictingDatabase, docs: Iterator[Database.DocCubbie],
                            hidden: Boolean, onlyAddToExistingTuples: Boolean = false,
                            minCellsPerRelation: Int = 2, minCellsPerTuple: Int = 2) {
    val cells = new ArrayBuffer[(Seq[Any], String, Double)]

    for (doc <- docs) {
      val relMentions = Database.KBDB.relMentions.query(_.docId(doc.name())).toArray
    //  logger.info("Processing %s with %d relation mentions".format(doc.name(), relMentions.size))
      for (rm <- relMentions) {
        val tuple = IndexedSeq(rm.arg1(), rm.arg2())
        val relation = rm.label()
        cells += ((tuple, relation, rm.gold.opt.getOrElse(1.0)))
      }
    }

    val relation2cells = cells.groupBy(_._2)
    val frequentRelCells = cells.filter(c => relation2cells.getOrElse(c._2, Seq.empty).size >= minCellsPerRelation)
    val tuple2cells = frequentRelCells.groupBy(_._1)
    val frequentCells = frequentRelCells.filter(c => tuple2cells.getOrElse(c._1, Seq.empty).size >= minCellsPerTuple)

    logger.info("%d frequent cells left of %d".format(frequentCells.size, cells.size))
    var added = 0
    for ((tuple, relation, target) <- frequentCells) {
      if ((!onlyAddToExistingTuples || spdb.getCells(tuple).size > 0) && spdb.getCell(relation, tuple).isEmpty) {
        spdb.addCell(relation, tuple, target, hidden, false, true)
        added += 1
      }
    }
    logger.info("%d relation mentions added".format(added))


  }

  def addRelationMentionsTo1(spdb: SelfPredictingDatabase, docs: Iterator[Database.DocCubbie],
                            hidden: Boolean, onlyAddToExistingTuples: Boolean = false,
                            minCellsPerRelation: Int = 2, minCellsPerTuple: Int = 2) : Seq[(Seq[Any], String, Double)] = {
    val cells = new ArrayBuffer[(Seq[Any], String, Double)]

    for (doc <- docs) {
      val relMentions = Database.KBDB.relMentions.query(_.docId(doc.name())).toArray
    //  logger.info("Processing %s with %d relation mentions".format(doc.name(), relMentions.size))
      for (rm <- relMentions) {
        val tuple = IndexedSeq(rm.arg1(), rm.arg2())
        val relation = rm.label()
        cells += ((tuple, relation, rm.gold.opt.getOrElse(1.0)))
      }
    }

    val relation2cells = cells.groupBy(_._2)
    val frequentRelCells = cells.filter(c => relation2cells.getOrElse(c._2, Seq.empty).size >= minCellsPerRelation)
    val tuple2cells = frequentRelCells.groupBy(_._1)
    val frequentCells = frequentRelCells.filter(c => tuple2cells.getOrElse(c._1, Seq.empty).size >= minCellsPerTuple)

    logger.info("%d frequent cells left of %d".format(frequentCells.size, cells.size))
    var added = 0
    for ((tuple, relation, target) <- frequentCells) {
      if ((!onlyAddToExistingTuples || spdb.getCells(tuple).size > 0) && spdb.getCell(relation, tuple).isEmpty) {
        spdb.addCell(relation, tuple, target, hidden, false, true)
        added += 1
      }
    }
    logger.info("%d relation mentions added".format(added))

    frequentCells
  }

  /**
   * This methods adds TAC slot cells to the matrix.
   * @param spdb the matrix to add cells to.
   * @param relations iterator of relation instances that should be added.
   * @param hidden Should the cells be observed or hidden.
   */
  def addTACGoldSlots(spdb: SelfPredictingDatabase, relations: Iterator[KBRelation], hidden: Boolean) {
    var totalSlotsLoaded = 0
    var totalSlotsAdded = 0
    for (rel <- relations) {
      //add cell but only if there is at least one existing cell for the given tuple
      val tuple = IndexedSeq(rel.arg1(), rel.arg2())
      val cells = spdb.getCells(tuple)
      if (cells.size > 0) {
        spdb.addCell(rel.label(), tuple, 1.0, hidden, false, true)
        
        //add tac label | srcner | dstner, todo

        
        totalSlotsAdded += 1
      }
      totalSlotsLoaded += 1
    }
    logger.info("Total slots loaded: " + totalSlotsLoaded)
    logger.info("Total slots added:  " + totalSlotsAdded)
  }

  /**
   * Creates cells for slots we haven't seen in the training data. When not hidden, and not yet in the matrix,
   * these cells will be considered as negative information.
   * @param spdb the matrix to add cells to.
   * @param entities the entities for which we add the slot cells for.
   * @param hidden should the cells be hidden (to be predicted) or observed (as negative data).
   */
  def addTACCandidateSlots(spdb: SelfPredictingDatabase, entities: Iterator[Entity], hidden: Boolean) {
    for (arg1 <- entities) {
      val relMentions = Database.KBDB.relMentions.query(_.arg1(arg1.id)).toArray
      for (rm <- relMentions) {
        val arg2 = rm.arg2()
        val tuple = IndexedSeq(arg1.id, arg2)
        for (slot <- Conf.targetSlots) {
          if (spdb.getCell(slot, tuple).isEmpty) {
            spdb.addCell(slot, tuple, 0.0, hidden, false, true)
          }
        }
      }
    }
  }

  //add negative surface pattern relations, implemented by Limin Yao
  //collect statistics of all surface pattern relations, do sampling as used in topic model
  def addNegativeSurfaceRelations(spdb : SelfPredictingDatabase, cells : Seq[(Seq[Any], String, Double)] ) {
    val relationStat = new HashMap[String, Int]
    val relations = new ArrayBuffer[String]
    for ((tuple, relation, target) <- cells; if(!relation.matches("[A-Z]+->[A-Z]+"))) {
      val ocount = relationStat.getOrElseUpdate(relation,0)
      relationStat(relation) = ocount + 1
      relations += relation
    }
    
    logger.info("Adding negative surface patterns!")
    val startTime = System.currentTimeMillis()
    //for each tuple, sample 10 times negative surface relations
    val tuple2cells = cells.groupBy(_._1)
    logger.info("Number of pairs:" + tuple2cells.size)
    var count = 0
    for(tuple <- tuple2cells.keys){
    //  logger.info("Add negative surface patterns for " + tuple.mkString("\t") )
      val tupleRelations = new HashSet[String]
      tupleRelations ++= spdb.getCells(tuple).map(_.relation)

      val total = tupleRelations.size * Conf.conf.getInt("pca.neg-features")
      for(iter <- 0 until  total){
        var sampledIndex = (math.random * relations.size ).toInt
        var sampledRelation = relations(sampledIndex)
        while(tupleRelations.contains(sampledRelation)){
          sampledIndex += 1
          sampledIndex = sampledIndex % relations.size
          sampledRelation = relations(sampledIndex)
        }
        if (spdb.getCell(sampledRelation, tuple).isEmpty)
          spdb.addCell(sampledRelation, tuple, 0.0, false, false, true)   //not sure about the last but two boolean value, Limin
      }
      count += 1
      if(count % 10000 == 0) logger.info("Processed tuples:" + count)
    }
    logger.info("Finish adding negative features within " + (System.currentTimeMillis() - startTime)/1000 + " seconds!")
  }

  def main(args: Array[String]) {
    Conf.add(args(0))
    runPCA()
  }


  /**
   * Writing out some debug output into the run-specific output directory.
   * @param spdb the matrix to debug.
   */
  def debugOutput(spdb: SelfPredictingDatabase) {
    val tuples = spdb.tuples.take(10000)
    val entityIds = tuples.flatMap(identity(_)).toSet.toSeq
    val entities = Database.CorefDB.entities.query(_.idsIn(entityIds)).toSeq.groupBy(_.id)

    spdb.debugTuples(tuples, new PrintStream(new File(Conf.outDir, "tuples.txt")),
      id => entities.get(id).flatMap(_.headOption).getOrElse("N/A").toString)
    spdb.debugModel(new PrintStream(new File(Conf.outDir, "model.txt")))
    val eval = spdb.evaluate(0.5)
    eval.debug(new PrintStream(new File(Conf.outDir, "pca-eval.txt")))
  }

  /**
   * Takes a matrix cell and determines its provenance.
   */
  trait Provenancer {
    def provenance(cell: SelfPredictingDatabase#Cell): Option[Mention]
  }

  object NoneProvenancer extends Provenancer {
    def provenance(cell: SelfPredictingDatabase#Cell) = None
  }

  /**
   * The simple provenancer uses as provenance the observed cell with highest probability under the current model.
   *
   * @param spdb the matrix to get cells from.
   * @param predictedCells the cells that have been predicted and need provenance.
   */
  class SimpleProvenancer(spdb: SelfPredictingDatabase,
                          predictedCells: Seq[SelfPredictingDatabase#Cell]) extends Provenancer {
    lazy val queryEntityIds = predictedCells.map(_.tuple(0)).toSet.toSeq
    //batch query to get all relation mentions for the query entities
    lazy val relevantRelMentions = Database.KBDB.relMentions.query(_.arg1.valuesIn(queryEntityIds)).toSeq
    //batch query to get all entity mentions corresponding to these
    lazy val relevantEntMentions = Database.CorefDB.linkedMentions.query(_.idsIn(relevantRelMentions.map(_.arg2Mention()).toSet.toSeq)).toSeq
    //mapping from tuple+relation to relation mentions
    lazy val tupleRel2mentions = relevantRelMentions.groupBy(rm => Seq(rm.arg1(), rm.arg2()) -> rm.label())
    //mapping from id to mention
    lazy val id2mentions = relevantEntMentions.groupBy(_.id)

    def provenance(cell: SelfPredictingDatabase#Cell): Option[Mention] = {
      //val provenanceCell = spdb.getCells(cell.tuple).find(_.target == 1.0)
      val provenanceCandidates = spdb.getCells(cell.tuple).filter(_.target == 1.0)
      val provenanceCell = if (provenanceCandidates.isEmpty) None else Some(provenanceCandidates.maxBy(cell => cell.update()))
      val provenanceRelMentions = provenanceCell.flatMap(c => tupleRel2mentions.get(c.tuple -> c.relation))
      val provenanceEntMentions = provenanceRelMentions.map(_.map(rm => id2mentions(rm.arg2Mention()).head))
      provenanceEntMentions.flatMap(_.headOption)
    }
  }

  /**
   * Writes out results in TAC format. This uses the set of TAC test entities in the database.
   * @param spdb the matrix.
   * @param threshold when to predict a slot as filled.
   */
  def writeOutTACResults(spdb: SelfPredictingDatabase, threshold: Double = 0.5) {
    logger.info("Writing out TAC slot predictions.")
    val tuples = spdb.tuples
    val entityIds = tuples.flatMap(identity(_)).toSet.toSeq
    val id2entities = Database.CorefDB.entities.query(_.idsIn(entityIds)).toSeq.groupBy(_.id)

    val predictedCells = new ArrayBuffer[SelfPredictingDatabase#Cell]

    //get predicted query cells
    for (tuple <- tuples) {
      if (id2entities(tuple(0)).head.tacTest.opt.getOrElse(false)) {
        for (cell <- spdb.getCells(tuple); if (Conf.targetSlots(cell.relation))) {
          cell.update()
          if (cell.predicted > threshold) {
            predictedCells += cell
          }
        }
      }
    }

    //the provenancer to use when writing out each row.
    val provenancer = Conf.conf.getString("eval.provenancer").toLowerCase match {
      case "simple" => new SimpleProvenancer(spdb, predictedCells)
      case _ => NoneProvenancer
    }



    //write out results to tab file
    val result = new PrintStream(new File(Conf.outDir, "tac-test.tab"))

   /* for ((cell, index) <- predictedCells.zipWithIndex) {
      val arg1 = id2entities(cell.tuple(0)).head
      val arg2 = id2entities(cell.tuple(1)).head

      //choose the best observed cell (for now: choose the first)
      val provenance = provenancer.provenance(cell)

      result.println(Seq(
        index,
        arg1.id,
        "IESL",
        cell.relation,
        provenance.map(_.docId()).getOrElse("NA"),
        provenance.map(_.charBegin()).getOrElse(0),
        provenance.map(_.charEnd()).getOrElse(0),
        arg2.canonical(),
        arg2.canonical(),
        cell.predicted
      ).mkString("\t"))
    } */

    val resultCells = new ArrayBuffer[String]
    for (cell <- predictedCells) {
      val arg1 = id2entities(cell.tuple(0)).head
      val arg2 = id2entities(cell.tuple(1)).head

      //choose the best observed cell (for now: choose the first)
      val provenance = provenancer.provenance(cell)

      resultCells += (Seq(
        arg1.id,
        "IESL",
        cell.relation,
        provenance.map(_.docId()).getOrElse("NA"),
        provenance.map(_.charBegin()).getOrElse(0),
        provenance.map(_.charEnd()).getOrElse(0),
        arg2.canonical(),
        arg2.canonical(),
        cell.predicted
      ).mkString("\t"))
    }
    result.println(resultCells.sortBy(_.toString).zipWithIndex.map(x => x._2 + "\t" + x._1).mkString("\n"))
    result.close()
  }

  /**
   * Prepares the matrix, runs the actual PCA algorithm, does some debug and evaluation output, and then writes out
   * the result in TAC format.
   */
  def runPCA() {
    logger.info("Preparing PCA")
    val spdb = new SelfPredictingDatabase()
    spdb.numComponents = Conf.conf.getInt("pca.rel-components")
    spdb.numArgComponents = Conf.conf.getInt("pca.arg-components")


    //train and test surface patterns. Test patterns are chosen based on hidden documents.
    //Note that hidden documents cannot be used for slot filling either.
    val numHiddenDocs = Conf.conf.getInt("eval.hidden-docs")
    val tupleCutoff = Conf.conf.getInt("pca.tuple-cutoff")
    val relCutoff = Conf.conf.getInt("pca.rel-cutoff")
//    addRelationMentionsTo(spdb, Database.CorefDB.documents.query(select = _.name.select).skip(numHiddenDocs),
//      hidden = false, minCellsPerTuple = tupleCutoff, minCellsPerRelation = relCutoff)
    val freqCells = addRelationMentionsTo1(spdb, Database.CorefDB.documents.query(select = _.name.select).skip(numHiddenDocs),
      hidden = false, minCellsPerTuple = tupleCutoff, minCellsPerRelation = relCutoff)

    //limin yao, not hiding any documents
//    addRelationMentionsTo(spdb, Database.CorefDB.documents.query(select = _.name.select).limit(numHiddenDocs),
//      hidden = true, minCellsPerRelation = 0, minCellsPerTuple = 0, onlyAddToExistingTuples = true)

    //todo: add negative surface patterns, only for training pairs
    addNegativeSurfaceRelations(spdb, freqCells)

    //KBP gold slots
    addTACGoldSlots(spdb, Database.KBDB.relations.query(_.tacTest(true)),  true)
    addTACGoldSlots(spdb, Database.KBDB.relations.query(_.tacTest(false)), false)

    //KBP candidate slots
    addTACCandidateSlots(spdb, Database.CorefDB.entities.query(_.tacTest(true)), hidden = true)
    addTACCandidateSlots(spdb, Database.CorefDB.entities.query(_.tacTest(false)), hidden = false)

    //add other distant supervision source relations, mainly Freebase relations
    addTACGoldSlots(spdb, Database.KBDB.relations.query(_.freebase(true)), false)

    //todo: extract job titles (should be happen earlier in the pipeline)
    //todo: add unary surface patterns
    //todo: incorporate Freebase relations (unary and binary)
    //todo: heuristically set TAC slots based on Freebase relations (used for learning)
    //todo: add PR constraints that imply TAC slots based on Freebase relations
    //todo: add PR constraints over Freebase relations (functionality, selectional preferences)

    //run PCA
    spdb.runLBFGS()

    //print out debug statistics
    debugOutput(spdb)

    //write out results
    writeOutTACResults(spdb)

  }
}
