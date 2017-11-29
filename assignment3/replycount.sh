# Constants
REDUCERS=5
TOP=10

# Remove the previous output
rm -rf output2 
rm -rf output
# Build and run the prog ram
mvn package
mvn exec:java -Dexec.mainClass="nl.uva.cpp.ReplyCount.ReplyCount" -Dexec.args="tweets.txt output $REDUCERS"
# Display the most popular repliers
echo "Top $TOP that reply the most:"
cat output/part-r-* | sort -r -k2 -t $'\t' | head -n $TOP
