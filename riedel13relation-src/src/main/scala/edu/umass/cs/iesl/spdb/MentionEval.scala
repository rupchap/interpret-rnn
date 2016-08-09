package edu.umass.cs.iesl.spdb

import Conf._
import cc.factorie.db.mongo.MongoCubbieImplicits._
import java.io.{File, PrintStream}
import cc.factorie.app.nlp.Document


/**
 * @author sriedel
 */
object MentionEval extends HasLogger {

  def fullDocName(name:String) = name

  def main(args: Array[String]) {
    add(args(0))
    evalMentions()
  }

  def evalMentions() {
    //check how many of the annotated relations could we possibly answer given the current mention linking
    val out = new PrintStream(new File(Conf.outDir, "mentioneval.txt"))
    val tacRels = Database.KBDB.tacRelations.toSeq
    val doc2Rels = tacRels.groupBy(_.tacProvenance().docId())
    var missedArg1 = 0
    var missedArg2 = 0
    var sameEntity = 0
    var multipleOverlap = 0
    var noFeats = 0
    var totalUniqueEntityCount = 0
    for ((docId, rels) <- doc2Rels) {
      logger.info("Processing doc " + docId)
      out.println("Processing doc " + docId)
      for (doc <- Database.CorefDB.documents.query(_.name(fullDocName(docId)))) {
        val fdoc = doc.fetchDocument
        val guessMentions = Database.CorefDB.linkedMentions.query(_.docId(doc.name())).toSeq
        out.println("LinkedMentions:" + guessMentions.size)
        val uniqueEntityCount = guessMentions.map(_.entity()).toSet.size
        val sent2Mentions = guessMentions.groupBy(m => FactorieTools.sentence(fdoc, m))
        val entities = Database.CorefDB.entities.query(_.idsIn(rels.map(_.arg1()))).toSeq.groupBy(_.id)
        totalUniqueEntityCount += uniqueEntityCount
        for (rel <- rels) {
          val arg1 = entities(rel.arg1()).head
          val mention = rel.tacProvenance()
          val overlap = guessMentions.filter(_.overlap(mention))

          val sentenceOpt = overlap.headOption.map(FactorieTools.sentence(fdoc, _))

          out.println("=============================")
          out.println("Doc ID:              " + docId)
          out.println("Query Entity:        " + arg1)
          out.println("Slot:                " + rel.label())
          out.println("Arg2 Phrase:         " + mention.phrase())
         // out.println("Arg2 Phrase (src):   " + FactorieTools.phrase(fdoc,mention))

          if (overlap.isEmpty) {
            missedArg2 += 1
          }
          if (overlap.size > 1) {
            multipleOverlap += 1
          }
          for (sentence <- sentenceOpt) {
            val otherMentions = sent2Mentions.getOrElse(sentence, Seq.empty)
            val otherArg1Mentions = otherMentions.filter(_.entity() == arg1.id)

            out.println("Sentence:            " + sentence.string)
            out.println("Mentions:            " + otherMentions.map(_.phrase()).mkString(","))
            //out.println("Mentions (src):      " + otherMentions.map(m => fdoc.string.substring(m.charBegin(),m.charEnd())).mkString(","))
            out.println("Overlaping mentions: " + overlap.map(_.phrase()).mkString(","))
            out.println("Arg1 Mentions:       " + otherArg1Mentions.map(_.phrase()).mkString(","))

            if (otherArg1Mentions.isEmpty) missedArg1 += 1
            else {
              val feats = FeatureExtractor.extractFeatures(otherArg1Mentions.head,overlap.head,fdoc)
              out.println("Feats:               " + feats.mkString(","))
              if (feats.size == 0) noFeats += 1
              if (otherArg1Mentions.head.entity() == overlap.head.entity()) sameEntity += 1
            }
          }

        }
      }
    }
    out.println("*************************")
    out.println("Relations:      " + tacRels.size)
    out.println("Missed Arg1:    " + missedArg1)
    out.println("Missed Arg2:    " + missedArg2)
    out.println("No Feats:       " + noFeats)
    out.println("Same Entities:  " + sameEntity)
    out.println("Mult. overlap:  " + multipleOverlap)
    out.println("Unique Entiies: " + totalUniqueEntityCount)
  }
}

object FactorieTools {
  def sentence(doc: Document, mention: Mention) = doc.tokens(mention.tokenBegin()).sentence
  def phrase(doc:Document, m:Mention) = doc.string.substring(m.charBegin(),m.charEnd())
}