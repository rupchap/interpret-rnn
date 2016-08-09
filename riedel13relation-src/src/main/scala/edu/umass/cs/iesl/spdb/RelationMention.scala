package edu.umass.cs.iesl.spdb

import cc.factorie.util.Cubbie


class RelationMention extends Cubbie {
  val label = StringSlot("label")
  val arg1 = RefSlot("arg1", () => new Entity)
  val arg2 = RefSlot("arg2", () => new Entity)
  val docId = StringSlot("docId")
  val arg1Mention = RefSlot("arg1Mention", () => new Mention)
  val arg2Mention = RefSlot("arg2Mention", () => new Mention)
  val gold = DoubleSlot("gold")

}

class EntityMention extends Cubbie {
  val label = StringSlot("label")
  val arg = RefSlot("arg", () => new Entity)
  val docId = StringSlot("docId")
  val argMention = RefSlot("argMention", () => new Mention)
  val gold = DoubleSlot("gold")

}



//todo: what's the type of arg1 and arg2
class KBRelation extends Cubbie {
  val arg1 = RefSlot("arg1", () => new Entity)
  val arg2 = RefSlot("arg2", () => new Entity)
  val label = StringSlot("label")
  val tacTest = BooleanSlot("tacTest")
  val tacProvenance = CubbieSlot("tacProvenance", () => new Mention)
  val freebase = BooleanSlot("freebase")
}



