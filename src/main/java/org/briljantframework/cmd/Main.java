package org.briljantframework.cmd;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.cli.*;
import org.apache.commons.math3.util.Pair;
import org.briljantframework.data.series.Series;
import org.briljantframework.mimir.classification.ClassifierValidator;
import org.briljantframework.mimir.classification.EnsembleEvaluator;
import org.briljantframework.mimir.classification.ProbabilityEstimator;
import org.briljantframework.mimir.classification.tree.ClassSet;
import org.briljantframework.mimir.classification.tree.TreeBranch;
import org.briljantframework.mimir.classification.tree.TreeLeaf;
import org.briljantframework.mimir.classification.tree.TreeNode;
import org.briljantframework.mimir.classification.tree.pattern.PatternDistance;
import org.briljantframework.mimir.classification.tree.pattern.PatternFactory;
import org.briljantframework.mimir.classification.tree.pattern.PatternTree;
import org.briljantframework.mimir.classification.tree.pattern.RandomPatternForest;
import org.briljantframework.mimir.data.ArrayInput;
import org.briljantframework.mimir.data.ArrayOutput;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Output;
import org.briljantframework.mimir.data.timeseries.TimeSeries;
import org.briljantframework.mimir.distance.EarlyAbandonSlidingDistance;
import org.briljantframework.mimir.evaluation.Result;
import org.briljantframework.mimir.shapelet.IndexSortedNormalizedShapelet;
import org.briljantframework.mimir.shapelet.Shapelet;

/**
 * A simple command line utility for running the random shapelet forest.
 */
public class Main {
  public static void main(String[] args) {
    // args = new String[] {"-n", "100", "-l", "0.025", "-u", "1", "-m",
    // "/Users/isak/mts_example/mts_data/CharacterTrajectories/train",
    // "/Users/isak/mts_example/mts_data/CharacterTrajectories/test"};
    Options options = new Options();

    options.addOption("n", "no-trees", true, "Number of trees");
    options.addOption("l", "lower", true, "Lower shapelet size (fraction of length, e.g, 0.05)");
    options.addOption("u", "upper", true, "Upper shapelet size (fraction of length, e.g, 0.8)");
    options.addOption("r", "sample", true, "Number of shapelets");
    options.addOption("p", "print-shapelets", false, "Print the shapelets of the forest");
    options.addOption("m", "multivariate", false, "The given dataset is in a multivariate format");
    options.addOption("c", "cv", true, "Cross-validation");

    CommandLineParser parser = new DefaultParser();
    try {
      CommandLine cmd = parser.parse(options, args);
      int noTrees = Integer.parseInt(cmd.getOptionValue("n", "100"));
      double lower = Double.parseDouble(cmd.getOptionValue("l", "0.025"));
      double upper = Double.parseDouble(cmd.getOptionValue("u", "1"));
      int r = Integer.parseInt(cmd.getOptionValue("r", "100"));
      boolean print = cmd.hasOption("p");

      List<String> files = cmd.getArgList();
      if (files == null || files.isEmpty()) {
        throw new RuntimeException("Training/testing data missing");
      }

      PatternFactory<MultivariateTimeSeries, MultivariateShapelet> patternFactory =
          new PatternFactory<MultivariateTimeSeries, MultivariateShapelet>() {

            /**
             * @param inputs the input dataset
             * @param classSet the inputs included in the current bootstrap.
             * @return a shapelet
             */
            public MultivariateShapelet createPattern(
                Input<? extends MultivariateTimeSeries> inputs, ClassSet classSet) {
              MultivariateTimeSeries mts =
                  inputs.get(classSet.getRandomSample().getRandomExample().getIndex());
              ThreadLocalRandom random = ThreadLocalRandom.current();
              int randomDim = random.nextInt(mts.dimensions());
              TimeSeries uts = mts.getDimension(randomDim);

              int MIN_LEN = 2, MAX_LEN = uts.size();
              if (lower > 0) {
                int frac = (int) Math.round(uts.size() * lower);
                MIN_LEN = frac > 2 ? frac : MIN_LEN;
              }

              if (upper > 0) {
                int frac = (int) Math.round(uts.size() * upper);
                MAX_LEN = frac > MIN_LEN ? frac : MAX_LEN;
              }

              int length = random.nextInt(MIN_LEN, MAX_LEN - 1);
              int start = random.nextInt(0, uts.size() - length - 1);
              // int length = random.nextInt(MIN_LEN, uts.size() - start);
              return new MultivariateShapelet(randomDim,
                  new IndexSortedNormalizedShapelet(start, length, uts));
            }
          };

      // Compute the minimum distance between the shapelet and the time series
      PatternDistance<MultivariateTimeSeries, MultivariateShapelet> patternDistance =
          new PatternDistance<MultivariateTimeSeries, MultivariateShapelet>() {
            private EarlyAbandonSlidingDistance distance = new EarlyAbandonSlidingDistance();

            public double computeDistance(MultivariateTimeSeries a, MultivariateShapelet b) {
              return distance.compute(a.getDimension(b.getDimension()), b.getShapelet());
            }
          };

      RandomPatternForest.Learner<MultivariateTimeSeries, Object> rsf =
          new RandomPatternForest.Learner<>(patternFactory, patternDistance, noTrees);
      rsf.set(PatternTree.PATTERN_COUNT, r);


      Pair<Input<MultivariateTimeSeries>, Output<Object>> train;
      ClassifierValidator<MultivariateTimeSeries, Object> validator;
      if (cmd.hasOption("c")) {
        Input<MultivariateTimeSeries> t = new ArrayInput<>();
        Output<Object> o = new ArrayOutput<>();
        if (cmd.hasOption("m")) {
          for (String file : files) {
            Pair<Input<MultivariateTimeSeries>, Output<Object>> data = readMtsData(file);
            t.addAll(data.getFirst());
            o.addAll(data.getSecond());
          }
        } else {
          for (String file : files) {
            Pair<Input<MultivariateTimeSeries>, Output<Object>> data = readData(file);
            t.addAll(data.getFirst());
            o.addAll(data.getSecond());
          }
        }
        train = new Pair<>(t, o);
        validator = ClassifierValidator.crossValidator(Integer.parseInt(cmd.getOptionValue("c", "10")));
      } else {
        Pair<Input<MultivariateTimeSeries>, Output<Object>> test;
        if (cmd.hasOption("m")) {
          train = readMtsData(files.get(0));
          test = readMtsData(files.get(1));
        } else {
          train = readData(files.get(0));
          test = readData(files.get(1));
        }
        validator = ClassifierValidator.holdoutValidator(test.getFirst(), test.getSecond());
      }
      validator.add(EnsembleEvaluator.getInstance());
      List<MultivariateShapelet> shapelets = new ArrayList<>();
      if (print) {
        validator.add(ctx -> {
          RandomPatternForest<MultivariateTimeSeries, Object> f =
              (RandomPatternForest<MultivariateTimeSeries, Object>) ctx.getPredictor();
          for (ProbabilityEstimator<MultivariateTimeSeries, Object> m : f.getEnsembleMembers()) {
            PatternTree<MultivariateTimeSeries, Object> t =
                (PatternTree<MultivariateTimeSeries, Object>) m;
            extractShapelets(shapelets, t.getRootNode());
          }
        });
      }

      Result<Object> result = validator.test(rsf, train.getFirst(), train.getSecond());

      if (print) {
        shapelets.sort((a, b) -> Integer.compare(a.getShapelet().size(), b.getShapelet().size()));
        for (int i = 0; i < shapelets.size(); i++) {
          System.out.print(i + "\t");
          Shapelet shapelet = shapelets.get(i).getShapelet();
          for (int j = 0; j < shapelet.size(); j++) {
            System.out.print(shapelet.getDouble(j) + " ");
          }
          System.out.println();
        }
      }

      System.out.println("Parameters");
      System.out.println("**********");
      for (Option o : cmd.getOptions()) {
        System.out.printf("%s:  %s\n", o.getLongOpt(), o.getValue("[default]"));
      }
      System.out.printf("Training data '%s'\n", files.get(0));
      System.out.printf("Testing data  '%s'\n", files.get(1));
      System.out.println(" ---- ---- ---- ---- ");

      System.out.println("\nResults");
      System.out.println("*******");
      Series measures = result.getMeasures().reduce(Series::mean);
      for (Object key : measures.index()) {
        System.out.printf("%s:  %.4f\n", key, measures.getDouble(key));
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

  private static void extractShapelets(List<MultivariateShapelet> shapelets,
      TreeNode<MultivariateTimeSeries, ?> node) {
    if (node instanceof TreeLeaf) {
      return;
    }

    @SuppressWarnings("unchecked")
    TreeBranch<MultivariateTimeSeries, PatternTree.Threshold<MultivariateShapelet>> b =
        (TreeBranch<MultivariateTimeSeries, PatternTree.Threshold<MultivariateShapelet>>) node;
    shapelets.add(b.getThreshold().getPattern());
    extractShapelets(shapelets, b.getLeft());
    extractShapelets(shapelets, b.getRight());
  }

  private static Pair<Input<MultivariateTimeSeries>, Output<Object>> readData(String filePath)
      throws IOException {
    // Construct the input and output variables
    Input<MultivariateTimeSeries> input = new ArrayInput<>();
    Output<Object> output = new ArrayOutput<>();

    // Read the file
    List<String> data = Files.readAllLines(Paths.get(filePath));
    Collections.shuffle(data, ThreadLocalRandom.current());
    for (String line : data) {
      String[] split = line.trim().split("\\s+");
      output.add(Double.parseDouble(split[0]));

      TimeSeries timeSeries = getTimeSeries(1, split);
      input.add(new MultivariateTimeSeries(timeSeries));
    }
    return new Pair<>(input, output);
  }

  private static TimeSeries getTimeSeries(int start, String[] split) {
    double[] ts = new double[split.length - start];
    for (int i = start; i < split.length; i++) {
      ts[i - start] = Double.parseDouble(split[i]);
    }
    // adding the same dimension twice - as an example. Each time-series should be distinct here.
    return TimeSeries.of(ts);
  }

  private static Pair<Input<MultivariateTimeSeries>, Output<Object>> readMtsData(String folder)
      throws IOException {
    Input<MultivariateTimeSeries> input = new ArrayInput<>();
    Output<Object> output = new ArrayOutput<>();

    String data = new String(Files.readAllBytes(Paths.get(folder, "classes.dat")));
    Collections.addAll(output, data.trim().split(","));

    List<Path> files = new ArrayList<>();
    Files.newDirectoryStream(Paths.get(folder, "data")).forEach(files::add);
    files.sort((a, b) -> getNameWithoutExt(a).compareTo(getNameWithoutExt(b)));

    for (Path exampleFile : files) {
      List<String> lines = Files.readAllLines(exampleFile);
      TimeSeries[] mts = new TimeSeries[lines.size()];
      for (int i = 0; i < lines.size(); i++) {
        String line = lines.get(i);
        String[] split = line.trim().split(",");
        mts[i] = getTimeSeries(0, split);
      }
      input.add(new MultivariateTimeSeries(mts));
    }
    return new Pair<>(input, output);
  }

  private static Integer getNameWithoutExt(Path a) {
    String name = a.getFileName().toString();
    return Integer.parseInt(name.split("\\.")[0]);
  }

}
