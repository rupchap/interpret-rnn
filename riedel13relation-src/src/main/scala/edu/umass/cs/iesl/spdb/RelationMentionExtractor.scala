package edu.umass.cs.iesl.spdb

import cc.factorie.db.mongo.MongoCubbieImplicits._
import cc.factorie.app.nlp.Document
import cc.factorie.app.nlp.parse.ParseTree
import cc.factorie.app.strings.Stopwords
import collection.mutable.{HashMap, ArrayBuffer}


/**
 * Goes through all documents and extracts relation mentions.
 */
object RelationMentionExtractor extends HasLogger {
  def main(args: Array[String]) {
    Conf.add(args(0))
    extractRelationMentions()
  }

  def extractRelationMentions() {

    for (doc <- Database.CorefDB.documents) {
      val relMentions = new ArrayBuffer[RelationMention]
      val mentions = Database.CorefDB.linkedMentions.query(_.docId(doc.name())).toArray
      logger.info("Extracting relation mentions from doc %s with %d mentions".format(doc.name(), mentions.size))
      val entityIds = mentions.map(_.entity())
      val id2entities = Database.CorefDB.entities.query(_.idsIn(entityIds)).toArray.groupBy(_.id)

      val facDoc = doc.fetchDocument

      for (source <- mentions) {
        val arg1 = id2entities(source.entity()).head

        for (dest <- mentions; if (facDoc.tokens(source.tokenBegin()).sentence.start == facDoc.tokens(dest.tokenBegin()).sentence.start //the same sentence
          && source.phrase() != dest.phrase())) {

          val arg2 = id2entities(dest.entity()).head

          val features = FeatureExtractor.extractFeatures(source, dest, facDoc)
          for (feat <- features) {
            //add provenance information to the database
            relMentions += new RelationMention().arg1(arg1.id).arg2(arg2.id).label(feat).
              arg1Mention(source.id).arg2Mention(dest.id).docId(doc.name())
          }

        }

      }
      Database.KBDB.relMentions ++= relMentions
    }
  }

  //not fixed yet
  def extractRelationMentions1() {
    val nerMap = new HashMap[String, String]()
    nerMap += "PER" -> "PERSON"
    nerMap += "ORG" -> "ORGANIZATION"

    val docColl = Database.CorefDB.coll("documents")
    for (doc <- Database.CorefDB.documents) {
//    val cursor = docColl.find()
//
//    cursor.options = 16
//    while(cursor.hasNext) {
//      doc = cursor.next()
      val relMentions = new ArrayBuffer[RelationMention]
      val mentions = Database.CorefDB.linkedMentions.query(_.docId(doc.name())).toArray
      logger.info("Extracting relation mentions from doc %s with %d mentions".format(doc.name(), mentions.size))
      val entityIds = mentions.map(_.entity())
      val id2entities = Database.CorefDB.entities.query(_.idsIn(entityIds)).toArray.groupBy(_.id)

      val facDoc = doc.fetchDocument

      for (source <- mentions) {
        val arg1 : Entity = id2entities(source.entity()).head

        for (dest <- mentions; if (facDoc.tokens(source.tokenBegin()).sentence.start == facDoc.tokens(dest.tokenBegin()).sentence.start //the same sentence
          && source.phrase() != dest.phrase())) {

          val arg2 : Entity = id2entities(dest.entity()).head

          var srcner = if(arg1.tacType.isDefined) nerMap(arg1.tacType()) else if(source.label.isDefined) source.label() else "NONE"
          var dstner = if(arg2.tacType.isDefined) nerMap(arg2.tacType()) else if(dest.label.isDefined) dest.label() else "NONE"

          //todo: only consider one arg as the tac queries
          if (arg1.tacType.isDefined || arg2.tacType.isDefined){
            val features = Features.extractFeatures(source, dest, facDoc, srcner, dstner)
            for (feat <- features) {
              //add provenance information to the database
              relMentions += new RelationMention().arg1(arg1.id).arg2(arg2.id).label(feat).
                arg1Mention(source.id).arg2Mention(dest.id).docId(doc.name())
            }
          }

        }

      }
      Database.KBDB.relMentions ++= relMentions
    }
  //  cursor.close()

  }

}

object EntityMentionExtractor extends HasLogger {
  def main(args: Array[String]) {
    Conf.add(args(0))
    extractEntityMentions()
  }

  def extractEntityMentions() {

    for (doc <- Database.CorefDB.documents) {
      val entMentions = new ArrayBuffer[EntityMention]
      val mentions = Database.CorefDB.linkedMentions.query(_.docId(doc.name())).toArray
      logger.info("Extracting entity mentions from doc %s with %d mentions".format(doc.name(), mentions.size))
      val entityIds = mentions.map(_.entity())
      val id2entities = Database.CorefDB.entities.query(_.idsIn(entityIds)).toArray.groupBy(_.id)
      val facDoc = doc.fetchDocument

      for (source <- mentions) {
        val arg1 = id2entities(source.entity()).head

        for (feat <- FeatureExtractor.extractFeatures(source,facDoc)) {
          entMentions += new EntityMention().arg(arg1.id).label(feat).
            argMention(source.id).docId(doc.name())

        }

      }

      Database.KBDB.entMentions ++= entMentions
    }
  }

}



object FeatureExtractor {

  def extractFeatures(source:Mention, doc:Document) = {
    val features = new ArrayBuffer[String]
    val sentence = FactorieTools.sentence(doc,source)
    val shead = EntityMentionUtils.headOfMention(source, sentence)
    features += source.label.opt.getOrElse("NA") + "/1"
    features

  }

  def extractFeatures(source: Mention, dest: Mention, doc: Document): Seq[String] = {
    val features = new ArrayBuffer[String]
    val sentence = FactorieTools.sentence(doc,source)
    val shead = EntityMentionUtils.headOfMention(source, sentence)
    val dhead = EntityMentionUtils.headOfMention(dest, sentence)
    val (rootOfPath, pathString, path) = DepUtils.find_path(shead, dhead, sentence.attr[ParseTree])
//    if (true) {
    if (math.abs(dest.tokenBegin() - source.tokenEnd()) < 20) {
      val (left,right,forward) = if (source.tokenBegin() < dest.tokenBegin()) (source,dest,true) else (dest,source,false)
      val lexContext = EntityMentionUtils.betweenLexContext(left, right, sentence)
      val lexPattern = lexContext.mkString(" ")
      features += (if (forward) "->" else "<-") + lexPattern
//      println(FactorieTools.phrase(doc,left) + "|" + lexPattern + "|" + FactorieTools.phrase(doc,left))
//      val posContext = EntityMentionUtils.betweenPOSContext(left, right, sentence)
      //todo: use pattern in "harvesting ..."
    }
//    println(pathString + " " + path.size)
    if (pathString != "junk" && pathString != "exception" && path.size < 6) {
      //create path feature
      features += pathString

      //create mention type pair
      features += source.label.opt.getOrElse("NA") + "->" + dest.label.opt.getOrElse("NA")

      //create surface string pattern if in close range
    }
    features.filter(_.length() > 0).filter(x => !(x.contains("javascript"))).map(_ + "/2")
  }


}
