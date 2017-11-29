package nl.uva.cpp.SentimentAnalysis;

import java.io.IOException;
import java.util.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

public class SentimentAnalysisReducer extends Reducer<Text, IntWritable, Text, Text> {

	@Override
	public void reduce(Text key, Iterable<IntWritable> values, Context context) 
			throws IOException, InterruptedException {
		int sum = 0;
		int count = 0;

		// Convert iterable to a list because we need to iterate over it twice
		List<Integer> list = new ArrayList<Integer>();
		for (IntWritable val : values) {
			int i = val.get();
			list.add(i);
		}

		// Calculate the mean sentiment
		for (Integer val : list) {
			sum += val;
			count++;
		}
		float mean = (float)sum / (float)count;
		double sd = 0;

		// Use the mean sentiment to calculate the standard deviation
		for (Integer val : list) {
			sd += Math.pow(val - mean, 2);
		}
		
		String value = String.valueOf(count) + "\t" + String.valueOf(mean) + "\t" + String.valueOf(sd);
		context.write(key, new Text(value));
	}
}
