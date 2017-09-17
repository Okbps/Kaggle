import util.Commons;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.io.*;

import static util.Globals.RESOURCES;

public class ApplicationDriver {
    public static void main(String[] args) throws Exception {
        ApplicationDriver training = new ApplicationDriver();
//        training.csvToArff("test");
//        training.csvToArff("train");
        training.buildModel();
    }

    private void buildModel() throws Exception {
        BufferedReader trainFile = Commons.readDataFile(RESOURCES + "arff/train-cleaned.arff");
        Instances trainData = new Instances(trainFile);
        trainData.setClassIndex(1);

        BufferedReader testFile = Commons.readDataFile(RESOURCES + "arff/test-cleaned.arff");
        Instances testData = new Instances(testFile);
        testData.setClassIndex(0);

        Classifiers cs = new Classifiers(trainData, testData);

        Classifier classifier = Classifiers.getStoredClassifier(RESOURCES+"models/weka-cleaned-logboost-xval4.model");

        cs.evaluateClassifier(classifier);

        Commons.classifyToCsv(classifier, testData);
    }



}
