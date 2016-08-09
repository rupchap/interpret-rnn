package edu.umass.cs.iesl.spdb

import cc.factorie.util.Cubbie
import com.mongodb.Mongo
import cc.factorie.db.mongo.MongoCubbieCollection

/**
 * @author sriedel
 */
class Mention extends Cubbie {
  val docId = StringSlot("docId")
  val charBegin = IntSlot("charBegin")
  val charEnd = IntSlot("charEnd")
  val tokenBegin = IntSlot("tokenBegin")
  val tokenEnd = IntSlot("tokenEnd")
  val phrase = StringSlot("phrase")
  val canonical = StringSlot("canonical")
  val label = StringSlot("label")   //ner label
  val entity = RefSlot("entity", () => new Entity)
  val tacMention = BooleanSlot("tacMention")

  def overlap(that:Mention) = charBegin() <= that.charBegin() && charEnd() >= that.charBegin() ||
    that.charBegin() <= charBegin() && that.charEnd() >= charBegin()
}

class Entity extends Cubbie {
  val freebaseId = StringSlot("freebaseId")
  //tacId, sf_id in annotation file
  val tacId = StringSlot("tacId")
  val tacType = StringSlot("tacType")
  val tacTest = BooleanSlot("tacTest")
  val wikipediaId = StringSlot("wikipediaId")
  val canonical = StringSlot("canonical")

}


class MentionTuple extends Cubbie {
  val mentions = CubbieListSlot("mentions", () => new Mention)
  val features = StringListSlot("features")
}

object CubbieExample {
  def main(args: Array[String]) {
    val mongoConn = new Mongo("localhost",27017)
    val mongoDB = mongoConn.getDB("tackbp-cubbie")   // web data for tac-kbp

    val mentions = new MongoCubbieCollection(mongoDB.getCollection("mentions"), () => new Mention, (m:Mention) => Seq(Seq(m.entity)))
    val entities = new MongoCubbieCollection(mongoDB.getCollection("entities"), () => new Entity)

    val e1 = new Entity
    e1.freebaseId := "fb001"

    val m1 = new Mention
    m1.charBegin := 14
    m1.charEnd := 20
    m1.docId := "testdoc"
    m1.entity ::= e1

    mentions += m1
    entities += e1

    val mentionsOfE1 = mentions.query(_.entity(e1.id))
    val mentionsOfDoc = mentions.query(_.docId("testdoc"))

    val tuple = new MentionTuple
    tuple.mentions := Seq(m1,m1)
    tuple.features := Seq("<-obj-take...")






  }
}