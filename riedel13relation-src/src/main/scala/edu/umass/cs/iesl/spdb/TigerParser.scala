package edu.umass.cs.iesl.spdb

/**
 * Created by IntelliJ IDEA.
 * User: lmyao
 * Date: 4/19/12
 * Time: 1:51 PM
 * To change this template use File | Settings | File Templates.
 */
import cc.factorie.app.nlp._
import cc.factorie.app.nlp.parse._
import dependency.DTree
import cc.refectorie.user.jzheng.coref.PosTag

object TigerParser extends Parser {

  var parserModel: String = "models/parse/ParserModel"
  private lazy val _parser = {
    print("Loading parser model..."); val p = parser.Parser.read(parserModel); print("DONE\n"); p
  }

  private def _parse(tokens: Array[String], tags: Array[String]): DTree =
    _parser.parse(new DTree(tokens, tags, new Array[Int](tokens.length), new Array[String](tokens.size)))

  def process(docs: Seq[Document]): Unit = docs.foreach(d => process(d))

  def process(doc: Document): Unit = {
    doc.sentences.foreach(s => parse(s))
    doc.sentences.foreach(s => s.foreach(t => assert(t.sentence eq s)))
  }

  override def parse(s: Sentence) = {
    val tree = new ParseTree(s)
    val tigerTree = _parse(s.tokens.map(_.string).toArray, s.tokens.map(_.attr[PosTag].value).toSeq.toArray)
    for (childIdx <- 0 until s.size) {
      val tigerToken = tigerTree.tokenAt(childIdx + 1)
      val parentIdx = tigerToken.getHead() - 1
      tree.setParent(childIdx, parentIdx)
      tree.label(childIdx).setCategory(tigerToken.getDeprel())(null)
    }
    s.attr += tree
  }

}
