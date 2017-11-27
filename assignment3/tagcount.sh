# Constants
REDUCERS=5
TOP=10

# Remove the previous output
rm -rf output
# Build and run the program
mvn package
mvn exec:java -Dexec.args="tweets.txt output/ $REDUCERS"
# Display the most popular hashtags
echo "Top $TOP most popular hashtags:"
cat output/part-r-* | sort -r -n -k2,2 -t $'\t' | head -n $TOP