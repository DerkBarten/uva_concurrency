rm -rf output
mvn package
mvn exec:java -Dexec.args="tweets.txt output/ 1"