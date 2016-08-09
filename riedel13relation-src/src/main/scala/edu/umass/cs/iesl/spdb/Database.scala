package edu.umass.cs.iesl.spdb

import com.mongodb.Mongo
import cc.factorie.app.nlp._
import cc.factorie.db.mongo.{LazyCubbieConverter, MongoCubbieCollection}

/**
 * @author sriedel
 */
object Database {

  type DocCubbie = DocumentCubbie[TokenCubbie with TokenPOSCubbie with TokenNerCubbie with TokenLemmaCubbie, SentenceCubbie with SentenceParseCubbie, TokenSpanCubbie]

  import Conf._

  trait DBEnv {
    def host: String

    def db: String

    def port: Int

    lazy val mongoConn = new Mongo(host, port)
    lazy val mongoDB = mongoConn.getDB(db)

    def coll(name: String) = mongoDB.getCollection(name)

    def dropDatabase() {
      mongoDB.dropDatabase()
    }

  }


  object CorefDB extends DBEnv {

    def host = conf.getString("source-data.mongo.host")

    def db = conf.getString("source-data.mongo.db")

    def port = conf.getInt("source-data.mongo.port")

    val documents = new MongoCubbieCollection[DocCubbie](coll("documents"),
      () => new DocumentCubbie(() => new TokenCubbie with TokenPOSCubbie with TokenNerCubbie with TokenLemmaCubbie,
        () => new SentenceCubbie with SentenceParseCubbie,
        () => new TokenSpanCubbie),
      (d: DocCubbie) => Seq(Seq(d.name))
    ) with LazyCubbieConverter[DocCubbie]

    val mentions = new MongoCubbieCollection(coll("mentions"), () => new Mention, (m: Mention) => Seq(Seq(m.entity), Seq(m.docId)))

    val linkedMentions = new MongoCubbieCollection(coll("linkedMentions"), () => new Mention,
      (m: Mention) => Seq(Seq(m.entity), Seq(m.docId)))

    val entities = new MongoCubbieCollection(coll("entities2"), () => new Entity,
      (e: Entity) => Seq(Seq(e.canonical), Seq(e.tacId), Seq(e.tacTest)))


  }

  object KBDB extends DBEnv {
    def host = conf.getString("kb-data.mongo.host")

    def db = conf.getString("kb-data.mongo.db")

    def port = conf.getInt("kb-data.mongo.port")

    val tacMentions = new MongoCubbieCollection(coll("tacMentions"), () => new Mention, (m: Mention) => Seq(Seq(m.entity), Seq(m.docId)))

    val tacTrainEntities = new MongoCubbieCollection(coll("tacTrainEntities"), () => new Entity,
      (e: Entity) => Seq(Seq(e.canonical), Seq(e.tacId)))

    val tacTestEntities = new MongoCubbieCollection(coll("tacTestEntities"), () => new Entity,
      (e: Entity) => Seq(Seq(e.canonical), Seq(e.tacId)))

    val relations = new MongoCubbieCollection(coll("relations"), () => new KBRelation)
    val tacRelations = new MongoCubbieCollection(coll("tacRelations"), () => new KBRelation)

    val relMentions = new MongoCubbieCollection(coll("relMentions"), () => new RelationMention,
      (rm:RelationMention) => Seq(Seq(rm.docId), Seq(rm.arg1,rm.arg2), Seq(rm.label)))

    val entMentions = new MongoCubbieCollection(coll("entMentions"), () => new EntityMention,
      (em:EntityMention) => Seq(Seq(em.docId), Seq(em.arg), Seq(em.label)))


  }


}
