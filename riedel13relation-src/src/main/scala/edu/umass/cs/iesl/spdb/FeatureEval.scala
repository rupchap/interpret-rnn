package edu.umass.cs.iesl.spdb

import Conf._
import cc.factorie.db.mongo.MongoCubbieImplicits._
import java.io.{File, PrintStream}
import cc.factorie.app.nlp.Document
import Database._
import KBDB._
import CorefDB._

/**
 * @author sriedel
 */
object FeatureEval extends HasLogger {
  def main(args: Array[String]) {
    add(args(0))
    evalFeatures()
  }

  def evalFeatures() {
    val out = new PrintStream(new File(Conf.outDir, "featureeval.txt"))

    for (doc <- documents.iterator.limit(10)) {
      val fdoc = doc.fetchDocument
      logger.info("Processing doc " + doc.name())
      val docRelMentions = relMentions.query(_.docId(doc.name())).toSeq
      val docMentions = linkedMentions.query(_.docId(doc.name())).toSeq
      val docEntities = entities.query(_.idsIn(docMentions.map(_.entity()).toSet.toSeq)).toSeq
      val arg12relMentions = docRelMentions.groupBy(_.arg1Mention())
      val id2Mentions = docMentions.groupBy(_.id)
      val id2Entities = docEntities.groupBy(_.id)

      out.println("==============")
      out.println(doc.string())
      for (sentence <- fdoc.sentences) {
        out.println("--------------")
        out.println("Sentence: " + sentence.string)
        val sentMentions = docMentions.filter(m => m.charBegin() >= sentence.head.stringStart &&
          m.charEnd() <= sentence.last.stringEnd).sortBy(_.charBegin())

        for (arg1 <- sentMentions) {
          out.println("*******")
          out.println("Arg1 Mention: " + arg1.phrase())
          for ((arg2id, rms) <- arg12relMentions.getOrElse(arg1.id, Seq.empty).groupBy(_.arg2Mention())) {
            val arg2 = id2Mentions(arg2id).head
            out.println("Arg2 Mention: " + arg2.phrase())
            for (rm <- rms) {
              out.println("Feat:         " + rm.label())
            }
          }
        }


      }

    }
  }

}
