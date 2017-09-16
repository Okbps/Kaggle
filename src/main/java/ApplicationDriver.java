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
        BufferedReader dataFile = Commons.readDataFile(RESOURCES + "arff/train.arff");
        Instances trainingData = new Instances(dataFile);
        trainingData.setClassIndex(1);

        Classifiers cs = new Classifiers(trainingData);

//        Classifier classifier = cs.getSmoClassifier();
//        Classifier classifier = cs.getMlpClassifier();
        Classifier classifier = Classifiers.getStoredClassifier(RESOURCES+"models/weka-mlp.model");

        // todo remove excess attributes
        cs.evaluateClassifier(classifier);

        Commons.classifyToCsv(classifier);
    }



}
