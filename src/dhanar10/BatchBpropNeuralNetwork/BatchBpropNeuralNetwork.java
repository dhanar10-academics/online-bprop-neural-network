package dhanar10.BatchBpropNeuralNetwork;

public class BatchBpropNeuralNetwork {
	public static final int HIDDEN_NEURON = 4;
	public static final double LEARNING_RATE = 0.7;
	public static final double TARGET_MSE = 0.001;
	public static final int MAX_EPOCH = 10000;

	public static void main(String[] args) {
		int status = 0;
		
		double yInput[] = new double[2];
		double yHidden[] = new double[HIDDEN_NEURON];
		double yOutput = 0;
		
		double wInputHidden[][] = new double[2][HIDDEN_NEURON];
		double wHiddenOutput[] = new double[HIDDEN_NEURON];
		
		int epoch = 0;
		
		double dTraining[][] = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}}; // XOR
		
		for (int i = 0; i < wInputHidden.length; i++) {
			for (int j = 0; j < wInputHidden[0].length; j++) {
				wInputHidden[i][j] = Math.random();
			}
		}
		
		for (int i = 0; i < wHiddenOutput.length; i++) {
			wHiddenOutput[i] = Math.random();
		}
		
		while (true) {
			double mse = 0;
			
			double yTarget = 0;
			
			double eHidden[] = new double[HIDDEN_NEURON];
			double eOutput = 0;
			
			double dwInputHidden[][] = new double[2][HIDDEN_NEURON];
			double dwHiddenOutput[] = new double[HIDDEN_NEURON];
			
			epoch++;
			
			for (int i = 0; i < dTraining.length; i++) {
				for (int j = 0; j < yInput.length; j++) {
					yInput[j] = dTraining[i][j];
				}
				
				yTarget = dTraining[i][dTraining[i].length - 1];
				
				for (int j = 0; j < yHidden.length; j++) {
					yHidden[j] = 0;
					
					for (int k = 0; k < yInput.length; k++) {
						yHidden[j] += yInput[k] * wInputHidden[k][j];
					}
					
					yHidden[j] = 1 / (1 + Math.pow(Math.E, -yHidden[j]));
				}
				
				yOutput = 0;
				
				for (int j = 0; j < yHidden.length; j++) {
					yOutput += yHidden[j] * wHiddenOutput[j];
				}
				
				yOutput = 1 / (1 + Math.pow(Math.E, -yOutput));
				
				eOutput = (yTarget  - yOutput) * yOutput * (1 - yOutput);
				
				for (int j = 0; j < yHidden.length; j++) {
					eHidden[j] = eOutput * wHiddenOutput[j] * yHidden[j] * (1 - yHidden[j]);
				}
				
				for (int j = 0; j < yHidden.length; j++) {
					for (int k = 0; k < yInput.length; k++) {
						dwInputHidden[k][j] += LEARNING_RATE * eHidden[j] * yInput[k];
					}
				}
				
				for (int j = 0; j < yHidden.length; j++) {
					dwHiddenOutput[j] += LEARNING_RATE * eOutput * yHidden[j];
				}
				
				mse += Math.pow(yTarget  - yOutput, 2);
			}
			
			for (int j = 0; j < yHidden.length; j++) {
				for (int k = 0; k < yInput.length; k++) {
					wInputHidden[k][j] += dwInputHidden[k][j];
				}
			}
			
			for (int j = 0; j < yHidden.length; j++) {
				wHiddenOutput[j] += dwHiddenOutput[j];
			}
			
			mse /= dTraining.length;
			
			System.out.println(epoch + "\t" + mse);
			
			if (mse < TARGET_MSE) {
				break;
			}
			
			if (epoch == MAX_EPOCH) {
				status = 1;
				break;
			}
		}
		
		System.out.println();
		
		for (int i = 0; i < dTraining.length; i++) {
			for (int j = 0; j < yInput.length; j++) {
				yInput[j] = dTraining[i][j];
			}
			
			for (int j = 0; j < yHidden.length; j++) {
				yHidden[j] = 0;
				
				for (int k = 0; k < yInput.length; k++) {
					yHidden[j] += yInput[k] * wInputHidden[k][j];
				}
				
				yHidden[j] = 1 / (1 + Math.pow(Math.E, -yHidden[j]));
			}
			
			yOutput = 0;
			
			for (int j = 0; j < yHidden.length; j++) {
				yOutput += yHidden[j] * wHiddenOutput[j];
			}
			
			yOutput = 1 / (1 + Math.pow(Math.E, -yOutput));
			
			System.out.println(yInput[0] + "\t" + yInput[1] + "\t" + yOutput);
		}
		
		System.exit(status);
	}
}
