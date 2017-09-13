import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import static util.Globals.RESOURCES;

public class ModelTraining {
    public static void main(String[] args) throws Exception {
        ModelTraining training = new ModelTraining();
        training.buildModel();
    }

    private void buildModel() throws Exception {
        BufferedReader dataFile = readDataFile(RESOURCES + "train.csv");
        Instances data = new Instances(dataFile);
        data.setClassIndex(1);

        // TODO make one file for train+test
    }

    private BufferedReader readDataFile(String filename){
        BufferedReader inputReader = null;
        try{
            inputReader = new BufferedReader(new FileReader(filename));
        }catch (FileNotFoundException e){
            e.printStackTrace();
        }
        return inputReader;
    }
}
