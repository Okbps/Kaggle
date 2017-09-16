import util.Commons;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.BufferedReader;
import java.io.FileInputStream;

import static util.Globals.RESOURCES;

public class Classifiers {
    private Instances trainingData;

    public Classifiers(Instances trainingInstances) {
        this.trainingData = trainingInstances;
    }

    public static Classifier getStoredClassifier(String fileName) throws Exception {
        return (Classifier) SerializationHelper.read(new FileInputStream(fileName));
    }

    public Classifier getSmoClassifier() throws Exception {
        Classifier smo = new SMO();
        smo.buildClassifier(trainingData);
        return smo;
    }

    public Classifier getMlpClassifier() throws Exception {
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        mlp.setLearningRate(0.1);
        mlp.setMomentum(0.2);
        mlp.setTrainingTime(2000);
        mlp.setHiddenLayers("3");

        mlp.buildClassifier(trainingData);

        return mlp;
    }

    public void evaluateClassifier(Classifier classifier) throws Exception {
        BufferedReader testFile = Commons.readDataFile(RESOURCES + "arff/test.arff");
        Instances testingData = new Instances(testFile);
        testingData.setClassIndex(1);

        Evaluation evaluation = new Evaluation(trainingData);

        evaluation.evaluateModel(classifier, testingData);
        System.out.println(evaluation.toSummaryString());
    }
}
