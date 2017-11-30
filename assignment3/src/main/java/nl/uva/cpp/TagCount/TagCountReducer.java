package nl.uva.cpp.TagCount;

import java.io.IOException;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

public class TagCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

	@Override
	public void reduce(Text key, Iterable<IntWritable> values, Context context) 
			throws IOException, InterruptedException {
		int sum = 0;
		int count = 0;
		for (IntWritable val : values) {
			sum += val.get();
			count++;
		}
		context.write(key, new IntWritable(sum));
	}
}
