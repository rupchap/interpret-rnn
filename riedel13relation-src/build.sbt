import com.typesafe.startscript.StartScriptPlugin

name := "spdb"

organization := "edu.umass.cs.iesl.spdb"

// The := method used in Name and Version is one of two fundamental methods.
// The other method is <<=
// All other initialization methods are implemented in terms of these.
version := "0.1-SNAPSHOT"

scalaVersion := "2.9.2"

resolvers += "conjars.org" at "http://conjars.org/repo"

scalacOptions ++= Seq("-unchecked","-deprecation")

resolvers ++= Seq(
    "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository",
    "Sameer" at "https://github.com/sameersingh/maven-repo/raw/master/snapshots",
    "TypeSafe" at "http://repo.typesafe.com/typesafe/releases/",
    "IESL third party" at "https://dev-iesl.cs.umass.edu/nexus/content/repositories/thirdparty/",
    "IESL snapshots" at "https://dev-iesl.cs.umass.edu/nexus/content/repositories/snapshots",
    "IESL releases" at "https://dev-iesl.cs.umass.edu/nexus/content/repositories/releases"
)

// Add multiple dependencies
libraryDependencies ++= Seq(
     "thirdparty" % "jgrapht-jdk1.6" % "0.8.2",
     "cc.factorie" % "factorie" % "0.10.2-SNAPSHOT",
     "junit" % "junit" % "4.8" % "test",
     "log4j" % "log4j" % "1.2.16",
     "trove" % "trove" % "1.0.2",
     "cc.refectorie.user.sameer" % "util" % "1.2",
     "org.riedelcastro.nurupo" %% "nurupo" % "0.1-SNAPSHOT",
     "xom" % "xom" % "1.2.5",
     "edu.stanford" % "stanford-corenlp-faust-models" % "2011-06-19",
     "edu.stanford" % "stanford-corenlp-faust" % "2011-07-22" ,
     "stax" % "stax" %  "1.2.0",
     "malt" % "malt" % "1.4.1-iesl",
     "com.typesafe" % "config" % "1.0.0" ,
     "cc.refectorie.user.jzheng.coref" % "uiuccoref_2.9.1" % "1.0-SNAPSHOT",
     "org.sameersingh.scalaplot" % "scalaplot" % "0.1-SNAPSHOT",
     "com.google.protobuf" % "protobuf-java" % "2.3.0"
 //    "com.jsuereth" %% "scala-arm" % "1.2"
    //  "net.sf.jwordnet" % "jwnl" % "1.4_rc3"
     )

mainClass := Some("edu.umass.cs.iesl.spdb.PcaDSRun")

fork in run := true

fork in runMain := true

connectInput in run := true

javaOptions in run += "-Xmx60G"

javaOptions in runMain += "-Xmx60G"


seq(StartScriptPlugin.startScriptForClassesSettings: _*)
