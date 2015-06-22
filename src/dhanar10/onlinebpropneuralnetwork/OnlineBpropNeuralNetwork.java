package dhanar10.onlinebpropneuralnetwork;

public class OnlineBpropNeuralNetwork {
	private double mse = 0;
	
	private double yInput[];
	private double yHidden[];
	private double yOutput[];
	
	private double wInputHidden[][];
	private double wHiddenOutput[][];

	public static void main(String[] args) {
		double data[][] = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}}; // XOR
		
		OnlineBpropNeuralNetwork bprop = new OnlineBpropNeuralNetwork(2, 4, 1);
		boolean success = bprop.train(data, 0.7, 0.001, 10000);
		
		System.out.println();
		
		for (int i = 0; i < data.length; i++) {
			double output[] = bprop.compute(data[i]);
			
			for (int j = 0; j < data[i].length - output.length; j++) {
				System.out.printf("%.2f%s", data[i][j], "\t");
			}
			
			for (int j = 0; j < output.length; j++) {
				System.out.printf("%.2f%s", output[j], j + 1 != output.length ? "\t" : "");
			}
			
			System.out.println();
		}
		
		System.exit(success ? 0 : 1);
	}
	
	public OnlineBpropNeuralNetwork(int input, int hidden, int output) {
		yInput = new double[input];
		yHidden = new double[hidden];
		yOutput = new double[output];
		
		wInputHidden = new double[input][hidden];
		wHiddenOutput = new double[hidden][output];
	}
	
	public boolean train(double data[][], double learningRate, double targetMse, double maxEpoch) {
		boolean success = true;
		
		int epoch = 0;
		
		for (int i = 0; i < wInputHidden.length; i++) {
			for (int j = 0; j < wInputHidden[0].length; j++) {
				wInputHidden[i][j] = Math.random() * 2 - 1;
			}
		}
		
		for (int i = 0; i < wHiddenOutput.length; i++) {
			for (int j = 0; j < wHiddenOutput[0].length; j++) {
				wHiddenOutput[i][j] = Math.random() * 2 - 1;
			}
		}
		
		// BEGIN Nguyen-Widrow
		
		double bInputHidden = 0.7 * Math.pow(yHidden.length, 1.0 / yInput.length);
		double nInputHidden = 0;
		
		for (int i = 0; i < wInputHidden.length; i++) {
			for (int j = 0; j < wInputHidden[0].length; j++) {
				nInputHidden += wInputHidden[i][j] * wInputHidden[i][j];
			}
		}
		
		nInputHidden = Math.sqrt(nInputHidden);
		
		for (int i = 0; i < wInputHidden.length; i++) {
			for (int j = 0; j < wInputHidden[0].length; j++) {
				wInputHidden[i][j] = bInputHidden * wInputHidden[i][j] / nInputHidden;
			}
		}
		
		double bHiddenOutput = 0.7 * Math.pow(yOutput.length, 1.0 / yHidden.length);
		double nHiddenOutput = 0;
		
		for (int i = 0; i < wHiddenOutput.length; i++) {
			for (int j = 0; j < wHiddenOutput[0].length; j++) {
				nHiddenOutput += wHiddenOutput[i][j] * wHiddenOutput[i][j];
			}
		}
		
		nHiddenOutput = Math.sqrt(nHiddenOutput);
		
		for (int i = 0; i < wHiddenOutput.length; i++) {
			for (int j = 0; j < wHiddenOutput[0].length; j++) {
				wHiddenOutput[i][j] = bHiddenOutput * wHiddenOutput[i][j] / nHiddenOutput;
			}
		}
		
		// END Nguyen-Widrow
		
		while (true) {
			epoch++;
			
			mse = 0;
			
			for (double[] d : data) {
				double yTarget[] = new double[yOutput.length];
				
				double eHidden[] = new double[yHidden.length];
				double eOutput[] = new double[yOutput.length];
				
				for (int i = 0; i < d.length; i++) {
					if (i < yInput.length) {
						yInput[i] = d[i];
					}
					else {
						yTarget[i - yInput.length] = d[i];
					}
				}
				
				this.compute(yInput);
				
				for (int i = 0; i < yOutput.length; i++) {
					eOutput[i] = (yTarget[i] - yOutput[i]) * dsigmoid(yOutput[i]);
				}
				
				for (int i = 0; i < yHidden.length; i++) {
					for (int j = 0; j < yOutput.length; j++) {
						eHidden[i] += eOutput[j] * wHiddenOutput[i][j];
					}
					
					eHidden[i] *= dsigmoid(yHidden[i]);
				}
				
				for (int j = 0; j < yHidden.length; j++) {
					for (int k = 0; k < yInput.length; k++) {
						wInputHidden[k][j] += learningRate * eHidden[j] * yInput[k];
					}
				}
				
				for (int j = 0; j < yOutput.length; j++) {
					for (int k = 0; k < yHidden.length; k++) {
						wHiddenOutput[k][j] += learningRate * eOutput[j] * yHidden[k];
					}
				}
				
				for (int j = 0; j < yOutput.length; j++) {
					mse += (yTarget[j] - yOutput[j]) * (yTarget[j] - yOutput[j]);
				}
			}
			
			mse /= data.length * data[0].length;
			
			System.out.println(epoch + "\t" + mse);
			
			if (mse < targetMse) {
				break;
			}
			
			if (epoch == maxEpoch) {
				success = false;;
				break;
			}
		}
		
		return success;
	}
	
	public double[] compute(double input[]) {
		for (int i = 0; i < yInput.length; i++) {
			yInput[i] = input[i];
		}
		
		for (int i = 0; i < yHidden.length; i++) {
			yHidden[i] = 0;
			
			for (int j = 0; j < yInput.length; j++) {
				yHidden[i] += yInput[j] * wInputHidden[j][i];
			}
			
			yHidden[i] = sigmoid(yHidden[i]);
		}
		
		for (int i = 0; i < yOutput.length; i++) {
			yOutput[i] = 0;
			
			for (int j = 0; j < yHidden.length; j++) {
				yOutput[i] += yHidden[j] * wHiddenOutput[j][i];
			}
			
			yOutput[i] = sigmoid(yOutput[i]);
		}
		
		return yOutput;
	}
	
	public double getMse() {
		return mse;
	}
	
	private double sigmoid(double x) {
		return 1 / (1 + Math.pow(Math.E, -x));
	}
	
	private double dsigmoid(double x) {
		return x * (1 - x);
	}
}
