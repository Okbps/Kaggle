import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.FileInputStream;


public class Classifiers {
    private Instances trainData;
    private Instances testData;

    public Classifiers(Instances trainData, Instances testData) {
        this.trainData = trainData;
        this.testData = testData;
    }

    public static Classifier getStoredClassifier(String fileName) throws Exception {
        return (Classifier) SerializationHelper.read(new FileInputStream(fileName));
    }

    public Classifier getSmoClassifier() throws Exception {
        Classifier smo = new SMO();
        smo.buildClassifier(trainData);
        return smo;
    }

    public Classifier getMlpClassifier() throws Exception {
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        mlp.setLearningRate(0.1);
        mlp.setMomentum(0.2);
        mlp.setTrainingTime(2000);
        mlp.setHiddenLayers("3");

        mlp.buildClassifier(trainData);

        return mlp;
    }

    public void evaluateClassifier(Classifier classifier) throws Exception {

        Evaluation evaluation = new Evaluation(trainData);
        evaluation.evaluateModel(classifier, testData);
        System.out.println(evaluation.toSummaryString());
    }
}
