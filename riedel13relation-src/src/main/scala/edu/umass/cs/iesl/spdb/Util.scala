package edu.umass.cs.iesl.spdb

import collection.mutable.{HashSet, HashMap, ArrayBuffer}
import io.Source
import java.io._

/**
 * Created by IntelliJ IDEA.
 * User: lmyao
 * Date: 4/11/12
 * Time: 11:52 AM
 * To change this template use File | Settings | File Templates.
 */

object Util {
  /**Recursively descend directory, returning a list of files. */
  def files(directory: File): Seq[File] = {
    if (!directory.exists) throw new Error("File " + directory + " does not exist")
    if (directory.isFile) return List(directory)
    val result = new ArrayBuffer[File]
    for (entry: File <- directory.listFiles) {
      if (entry.isFile) result += entry
      else if (entry.isDirectory) result ++= files(entry)
    }
    result
  }

  //load relation triples, per:origin Limin Yao China
  def loadTripleHash(file: String, toLowerCase : Boolean = false): HashMap[String, String] = {
    val res = new HashMap[String, String]
    val source = Source.fromFile(file)
    for (line <- source.getLines) {
      val fields = line.split("\t")
      if(toLowerCase)
        res += (fields(1).toLowerCase + "\t" + fields(2).toLowerCase) -> fields(0)
      else 
        res += fields(1) + "\t" + fields(2) -> fields(0)
    }
    res
  }

  //load relation triples, per:origin Limin Yao China
  def loadTripleHash1(file: String, toLowerCase : Boolean = false): HashMap[String, String] = {
    val res = new HashMap[String, String]
    val source = Source.fromFile(file)
    for (line <- source.getLines) {
      val fields = line.split("\t")
      res += fields(0) + "\t" + fields(1) -> fields.drop(2).mkString("\t")
    }
    res
  }

  //allow one entity pair to have multiple labels , todo: debug this, find out multiple label tuples
  def loadTripleMultiHash(file: String, toLowerCase : Boolean = false): HashMap[String, HashSet[String]] = {
    val res = new HashMap[String, HashSet[String]]
    val source = Source.fromFile(file)
    for (line <- source.getLines) {
      val fields = line.split("\t")
      if(toLowerCase)
        res.getOrElseUpdate (fields(1).toLowerCase + "\t" + fields(2).toLowerCase,new HashSet[String]) += fields(0)
      else
        res.getOrElseUpdate (fields(1) + "\t" + fields(2), new HashSet[String]) += fields(0)
    }
    res
  }

  def loadHash(file: String): HashMap[String, String] = {
    val res = new HashMap[String, String]
    val source = Source.fromFile(file)
    for (line <- source.getLines) {
      val fields = line.split("\t")
      if(fields.length > 1)
        res += fields(0) -> fields(1)
    }
    res
  }

  def loadRevHash(file: String): HashMap[String, String] = {
    val res = new HashMap[String, String]
    val source = Source.fromFile(file)
    for (line <- source.getLines) {
      val fields = line.split("\t")
      res += fields(1) -> fields(0)
    }
    res
  }

  def loadHashSet(file: String): HashSet[String] = {
    println("Load from " + file)
    val res = new HashSet[String]
    val fstream = new FileInputStream(file)
    val in = new DataInputStream(fstream)
    val br = new BufferedReader(new InputStreamReader(in))
    var strLine : String = ""
    strLine = br.readLine
    var count = 0
    do {
      res.add(strLine)
      count += 1
      strLine = br.readLine
    }while(strLine != null)
    br.close
    in.close
    res
  }

  def loadHashSetLowercase(file: String): HashSet[String] = {
    val res = new HashSet[String]
    val source = Source.fromFile(file)
    for (line <- source.getLines) {
      res.add(line.toLowerCase)
    }
    res
  }
  
  def loadMultiMap(file : String) : HashMap[String, ArrayBuffer[String]] = {
    val res = new HashMap[String, ArrayBuffer[String]]()
    val source = Source.fromFile(file)
    for (line <- source.getLines()){
      val fields = line.split(" ")
      val seq = res.getOrElseUpdate(fields(1),new ArrayBuffer[String])
      seq += fields(0)
    }
    res
  }
  
  //check whether a file contains any of the keywords
  def checkFile(file : File, keywords : Seq[String]) : Boolean = {
    var str = readFile(file)
    var res = false
    for(keyword <- keywords){
      if(str.contains(keyword) || str.replaceAll ("\n", "").contains(keyword)){
    //    println(keyword)
        res = true
      }
    }
    res
  }

  def checkOverlap(file : File, keywords : HashSet[String]) : Boolean = {
    var res = false
    val source = Source.fromFile(file)
    for (line <- source.getLines()){
      if (keywords.contains(line)){
        println(line)
        return true
      }
    }
    res
  }

  def checkSource(txt : String, keywords : Seq[String]) : Boolean = {
    var res = false
    if(txt.length() == 0) return res
    for(keyword <- keywords){
      if(txt.contains(keyword) || txt.replaceAll ("\n", "").contains(keyword) ) res = true
    }
    res
  }

  def readFile(  file : File) : String =  {
    val reader = new BufferedReader( new FileReader (file))
    var line : String = null
    val stringBuilder = new StringBuilder()
    val ls = System.getProperty("line.separator");
    line = reader.readLine()
    while( line != null ) {
      stringBuilder.append( line )
      stringBuilder.append( ls )
      line = reader.readLine()
    }
    reader.close()
    if(stringBuilder.toString().length() > 80000) {
      var res = stringBuilder.toString()
      if(res.contains("<POST>")) {
        val splitpos = res.indexOf("</POST>")
        res = res.substring(0,splitpos+7)
        return res
      }
      else{
        println("Long doc:" + file.getCanonicalPath)
        return ""
      }
    }
    else return stringBuilder.toString()
  }
  
  def abbr(str : String) : String = {
    var res = ""
    val fields = str.split("[-\\s]")
    if(fields.size < 3) return "#None#"
    if(fields.size == 3 && fields(1).length() < 3) return "#None#"
    for(word <- fields){
      if(word.matches("^[A-Z].*")) res += word.substring(0,1)
    }
    res
  }

  def main(args: Array[String]) {
    val file = new File("/Users/lmyao/Documents/ie_ner/unsuprel/tackbp/eng-NG-31-100675-11559479.sgm")
    val keywords = loadHashSet("/Users/lmyao/Documents/ie_ner/unsuprel/tackbp/tac_queries.txt")
    println(checkFile(file,keywords.toSeq))
  }
}