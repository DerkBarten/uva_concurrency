# Constants
REDUCERS=5
TOP=50

# Remove the previous output
rm -rf output
# Build and run the program
mvn package
mvn exec:java  -Dexec.mainClass="nl.uva.cpp.SentimentAnalysis.SentimentAnalysis" -Dexec.args="tweets.txt output/ $REDUCERS"
# Display the most popular hashtags
echo ""
echo "Highest average sentiment:"
echo "Tag - Count - Average - Standard deviation"
cat output/part-r-* | sort -r -n -k3,3 -t $'\t' | head -n $TOP

echo ""
echo "Lowest average sentiment:"
echo "Tag - Count - Average - Standard deviation"
cat output/part-r-* | sort -n -k3,3 -t $'\t' | head -n $TOP