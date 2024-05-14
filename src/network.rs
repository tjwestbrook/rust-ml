use {
	super::{
		matrix::*,
		activations::*,
	},
	std::io::{Read, Write},
};

pub struct MultiLayerPerceptron {
	layers: Vec<usize>,
	weights: Vec<Matrix>,
	biases: Vec<Matrix>,
	data: Vec<Matrix>,
	activation: Activation,
	learning_rate: f64,
}

#[derive(serde::Serialize, serde::Deserialize, )]
struct SaveData {
	weights: Vec<Vec<Vec<f64>>>,
	biases: Vec<Vec<Vec<f64>>>,
}

impl MultiLayerPerceptron {
	pub fn new(
		layers: Vec<usize>, learning_rate: f64, activation: Activation,
	) -> Self {
		let (mut weights, mut biases) = (vec![], vec![]);
		for i in 0..layers.len() - 1 {
			weights.push(Matrix::random( layers[i], layers[i + 1]));
			biases.push(Matrix::random( 1, layers[i + 1]));
		}
		Self {
			layers,
			weights,
			biases,
			data: vec![],
			activation,
			learning_rate,
		}
	}

	pub fn train_and_test(
		&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16, success_rate: f64,
	) {
		self.train(inputs.clone(), targets.clone(), epochs, false);
		self.test(inputs, targets, success_rate, true);
	}

	pub fn train(
		&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16, print: bool,
	) {
		for i in 1..=epochs {
			if print && (epochs < 100 || i % (epochs / 100) == 0) { println!("Epoch {} of {}", i, epochs) }
			for j in 0..inputs.len() {
				let output = self.feed_forward(inputs[j].clone());
				self.back_propagate(output, targets[j].clone());
			}
		}
	}

	pub fn test(
		&self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, success_rate: f64, assert: bool,
	) -> f64 {
		if inputs.len() != targets.len() {
			panic!("Invalid inputs & targets lengths!")
		} else if targets[targets.len() - 1].len() != self.layers[self.layers.len() - 1] {
			panic!("Invalid target length for inference!")
		};
		
		for k in 0..self.weights.len() {
			println!("\nWEIGHTS at LAYER[{}] : {:?}\n", k, self.weights[k].clone())
		}
		
		let mut success_count = vec![0; inputs.len()];
		for i in 0..success_count.len() {
			let prediction = self.inference(inputs[i].clone());
			for j in (0..prediction.len()).into_iter() {
				if targets[i][j].clone() == prediction[j].round() { success_count[i] += 1 }
				println!("{} - Expected: {}\nPrediction[{}]: {}", i, targets[i][j], j, prediction[j]);
			}
			println!("Success count of {:?}: {} out of {}", inputs[i], success_count[i], prediction.len());
		}
		
		let total_success_count = success_count.iter().fold(0, |acc, x| acc + x);
		let actual_success_rate = total_success_count as f64 / (targets[0].len() * success_count.len()) as f64;
		println!("\nACTUAL SUCCESS RATE: {}\n", actual_success_rate);
		if assert { assert!(actual_success_rate >= success_rate) };
		actual_success_rate
	}

	pub fn inference(&self, input: Vec<f64>) -> Vec<f64> {
		if input.len() != self.layers[0] { panic!("Invalid input length for inference!") }
		let mut current = Matrix::from(vec![input]);
		(0..self.layers.len() - 1).into_iter().for_each(|i| {
			current *= self.weights[i].clone();
			current += self.biases[i].clone();
			current = current.map(self.activation.function);
		});
		current.data[0].to_owned()
	}

	fn feed_forward(&mut self, input: Vec<f64>) -> Vec<f64> {
		if input.len() != self.layers[0] { panic!("Invalid input length!") }
		
		let mut current = Matrix::from(vec![input]);
		self.data = vec![current.clone()];
		
		(0..self.layers.len() - 1).into_iter().for_each(|i| {
			current *= self.weights[i].clone();
			current += self.biases[i].clone();
			current = current.map(self.activation.function);
			self.data.push(current.clone());
		});
		
		current.data[0].to_owned()
	}

	fn back_propagate(&mut self, output: Vec<f64>, target: Vec<f64>) {
		if target.len() != self.layers[self.layers.len() - 1] { panic!("Invalid target length!") }
		
		let parsed = Matrix::from(vec![output]);
		let mut gradients = parsed.map(self.activation.derivative);
		let mut errors = Matrix::from(vec![target]) - parsed;
		
		(0..self.layers.len() - 1).rev().for_each(|i| {
			gradients = gradients.hadamard(&errors).map(|x| x * self.learning_rate);

			self.weights[i] += self.data[i].transpose() * gradients.clone();
			self.biases[i] += gradients.clone();

			errors *= self.weights[i].transpose();
			gradients = self.data[i].map(self.activation.derivative);
		});
	}

	pub fn save(&self, file: String) {
		let mut file = std::fs::File::create(file).expect("Unable to create save file!");
		file.write_all(serde_json::json!({
			"weights": self.weights.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f64>>>>(),
			"biases": self.biases.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f64>>>>()
		}).to_string().as_bytes()).expect("Unable to write to save file!");
	}

	pub fn load(&mut self, file: String) {
		let mut file = std::fs::File::open(file).expect("Unable to open save file!");
		let mut buffer = String::new();
		file.read_to_string(&mut buffer).expect("Unable to read save file!");
		let save_data: SaveData = serde_json::from_str(&buffer).expect("Unable to serialize save data!");
		for i in 0..self.layers.len() - 1 {
			self.weights.push(Matrix::from(save_data.weights[i].clone()));
			self.biases.push(Matrix::from(save_data.biases[i].clone()));
		}
	}
}

// #[cfg(test)]
pub mod test {
  use super::*;
	
	#[cfg(test)]
	fn not() {
		let inputs = vec![vec![0.0], vec![1.0]];
		let targets= vec![vec![1.0], vec![0.0]];
		let (layers, learning_rate, activation) = (vec![1, 1], 0.5, SIGMOID);
		let mut network = MultiLayerPerceptron::new(layers, learning_rate, activation);
		network.train_and_test(inputs, targets, 1000, 1.0);
	}

	#[cfg(test)]
	fn and() {
		let inputs = vec![	vec![0.; 2],vec![0., 1.], vec![1., 0.], vec![1.; 2]];
		let targets = vec![vec![0.],		vec![0.],			vec![0.],			vec![1.]];
		let (layers, learning_rate, activation) = (vec![2, 1], 0.5, SIGMOID);
		let mut network = MultiLayerPerceptron::new(layers, learning_rate, activation);
		network.train_and_test(inputs, targets, 1000, 1.);
	}

	#[cfg(test)]
	fn nand() {
		let inputs = vec![	vec![0.; 2],vec![0., 1.], vec![1., 0.], vec![1.; 2]];
		let targets = vec![vec![1.],		vec![1.],			vec![1.],			vec![0.]];
		let (layers, learning_rate, activation) = (vec![2, 1], 0.5, SIGMOID);
		let mut network = MultiLayerPerceptron::new(layers, learning_rate, activation);
		network.train_and_test(inputs, targets, 1000, 1.);
	}

	#[cfg(test)]
	fn or() {
		let inputs = vec![	vec![0.; 2],vec![0., 1.], vec![1., 0.], vec![1.; 2]];
		let targets = vec![vec![0.],		vec![1.],			vec![1.],			vec![1.]];
		let (layers, learning_rate, activation) = (vec![2, 1], 0.5, SIGMOID);
		let mut network = MultiLayerPerceptron::new(layers, learning_rate, activation);
		network.train_and_test(inputs, targets, 1000, 1.);
	}

	#[cfg(test)]
	fn nor() {
		let inputs = vec![	vec![0.; 2],vec![0., 1.], vec![1., 0.], vec![1.; 2]];
		let targets = vec![vec![1.],		vec![0.],			vec![0.],			vec![0.]];
		let (layers, learning_rate, activation) = (vec![2, 1], 0.5, SIGMOID);
		let mut network = MultiLayerPerceptron::new(layers, learning_rate, activation);
		network.train_and_test(inputs, targets, 1000, 1.);
	}

	// #[cfg(test)]
	pub fn xor() {
		let inputs = vec![	vec![0.; 2],vec![0., 1.], vec![1., 0.], vec![1.; 2]];
		let targets = vec![vec![0.],		vec![1.],			vec![1.],			vec![0.]];
		let (layers, learning_rate, activation) = (vec![2, 3, 1], 0.5, SIGMOID);
		let mut network = MultiLayerPerceptron::new(layers, learning_rate, activation);
		network.train_and_test(inputs, targets, 1000, 1.);
	}

	#[cfg(test)]
	fn xnor() {
		let inputs = vec![	vec![0.; 2],vec![0., 1.], vec![1., 0.], vec![1.; 2]];
		let targets = vec![vec![0.],		vec![1.],			vec![1.],			vec![0.]];
		let (layers, learning_rate, activation) = (vec![2, 3, 1], 0.5, SIGMOID);
		let mut network = MultiLayerPerceptron::new(layers, learning_rate, activation);
		network.train_and_test(inputs, targets, 1000, 1.);
	}

	#[cfg(test)]
	fn addition() {
		let (layers, learning_rate, activation) = (vec![3, 4, 1], 0.5, SIGMOID);
		let mut network = MultiLayerPerceptron::new(layers, learning_rate, activation);
		let (inputs, targets) = (
			vec![	vec![1., 2., 3.], vec![1., 0., 1.], vec![1., 1., 2.], vec![0., 1., 1.],
						vec![1., 2., 2.], vec![1., 0., 2.], vec![1., 1., 0.], vec![0., 1., 0.]],
			vec![	vec![1.],					vec![1.],					vec![1.],					vec![1.],
						vec![0.],					vec![0.],					vec![0.],					vec![0.]]
		);
		network.train_and_test(inputs, targets, 2000, 1.);
		assert_eq!(1., network.inference(vec![0., 3., 3.,])[0].round());
		assert_eq!(0., network.inference(vec![1., 2., 4.,])[0].round());
	}

	#[cfg(test)]
	fn subtraction() {
		let (layers, learning_rate, activation) = (vec![3, 4, 1], 0.5, SIGMOID);
		let mut network = MultiLayerPerceptron::new(layers, learning_rate, activation);
		let (inputs, targets) = (
			vec![	vec![1., 1., 0.], vec![3., 2., 1.], vec![3., 1., 2.], vec![1., 0., 1.],
						vec![3., 1., 0.], vec![3., 3., 1.], vec![1., 0., 0.], vec![1., 1., 1.]],
			vec![	vec![1.],					vec![1.],					vec![1.],					vec![1.],
						vec![0.],					vec![0.],					vec![0.],					vec![0.]]
		);
		network.train_and_test(inputs, targets, 2000, 1.);
	}
}