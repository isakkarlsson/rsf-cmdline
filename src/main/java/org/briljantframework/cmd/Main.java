package org.briljantframework.cmd;

import java.io.File;
import java.io.FileInputStream;
import java.util.List;

import org.apache.commons.cli.*;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataseries.DataSeriesCollection;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.MatlabDatasetReader;
import org.briljantframework.mimir.classification.ClassifierValidator;
import org.briljantframework.mimir.classification.EnsembleEvaluator;
import org.briljantframework.mimir.classification.RandomShapeletForest;
import org.briljantframework.mimir.evaluation.Result;

/**
 * A simple command line utility for running the random shapelet forest.
 */
public class Main {
  public static void main(String[] args) {
    Options options = new Options();

    options.addOption("n", "no-trees", true, "Number of trees");
    options.addOption("l", "lower", true, "Lower shapelet size (fraction of length, e.g, 0.05)");
    options.addOption("u", "upper", true, "Upper shapelet size (fraction of length, e.g, 0.8)");
    options.addOption("r", "sample", true, "Number of shapelets");
    CommandLineParser parser = new DefaultParser();
    try {
      CommandLine cmd = parser.parse(options, args);
      int noTrees = Integer.parseInt(cmd.getOptionValue("n", "100"));
      double lower = Double.parseDouble(cmd.getOptionValue("l", "0.025"));
      double upper = Double.parseDouble(cmd.getOptionValue("u", "1"));
      int r = Integer.parseInt(cmd.getOptionValue("r", "100"));

      List<String> files = cmd.getArgList();
      if (files == null || files.isEmpty()) {
        throw new RuntimeException("Training/testing data missing");
      }

      File trainFile = new File(files.get(0));
      File testFile = new File(files.get(1));
      RandomShapeletForest.Configurator configurator =
          new RandomShapeletForest.Configurator(noTrees);
      configurator.setLowerLength(lower);
      configurator.setUpperLength(upper);
      configurator.setMaximumShapelets(r);

      RandomShapeletForest.Learner rsf = configurator.configure();

      DataSeriesCollection.Builder trainDataBuilder =
          new DataSeriesCollection.Builder(double.class);
      MatlabDatasetReader trainReader = new MatlabDatasetReader(new FileInputStream(trainFile));
      trainDataBuilder.readAll(trainReader);

      DataSeriesCollection.Builder testDataBuilder = new DataSeriesCollection.Builder(double.class);
      MatlabDatasetReader testReader = new MatlabDatasetReader(new FileInputStream(testFile));
      testDataBuilder.readAll(testReader);

      DataFrame train = trainDataBuilder.build();
      DataFrame test = testDataBuilder.build();

      ClassifierValidator<RandomShapeletForest> validator =
          ClassifierValidator.holdoutValidator(test.loc().drop(0), test.loc().get(0));
      validator.add(EnsembleEvaluator.INSTANCE);

      Result result = validator.test(rsf, train.loc().drop(0), train.loc().get(0));

      System.out.println("Parameters");
      System.out.println("**********");
      for (Option o : cmd.getOptions()) {
        System.out.printf("%s:  %s\n", o.getLongOpt(), o.getValue("[default]"));
      }
      System.out.printf("Training data '%s'\n", files.get(0));
      System.out.printf("Testing data  '%s'\n", files.get(0));
      System.out.println(" ---- ---- ---- ---- ");

      System.out.println("\nResults");
      System.out.println("*******");
      Vector measures = result.getMeasures().mean();
      for (Object key : measures.getIndex()) {
        System.out.printf("%s:  %.2f\n", key, measures.getAsDouble(key));
      }
      System.out.println(" ---- ---- ---- ---- ");
      System.out.printf("Runtime (training)  %.2f ms\n", result.getFitTime());
      System.out.printf("Runtime (testing)   %.2f ms\n", result.getPredictTime());
    } catch (Exception e) {
      HelpFormatter formatter = new HelpFormatter();
      e.printStackTrace();
      formatter.printHelp("rsfcmd.jar [OPTIONS] trainFile testFile", options);
    }
  }
}
