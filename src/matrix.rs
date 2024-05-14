use {
	rand::{thread_rng, Rng},
	std::{
		ops::{Add, AddAssign, Sub, Mul, MulAssign},
		fmt::{Debug, Formatter, Result},
		iter::Sum,
	},
};

#[derive(Clone, PartialEq, )]
pub struct Matrix<T = f64> {
	rows: usize,
	cols: usize,
	pub data: Vec<Vec<T>>,
}

impl<T: Copy> Matrix<T> {
	fn confirm_dimension_operation(&self, rhs: &Self, op: &str) {
		if match op {
			"+" | "+=" | "-" | "o" => (self.rows, self.cols) != (rhs.rows, rhs.cols),
			"*=" | "*" => self.cols != rhs.rows,
			_ => true
		} {
			panic!("Attempted to {} with matrices of incorrect dimensions! {} x {} & {} x {}",
				match op {
					"+" 	=> "add",
					"+=" 	=> "add and assign",
					"-" 	=> "subtract",
					"o" 	=> "hadamard multiply",
					"*" 	=> "dot multiply",
					"*=" 	=> "dot multiply and assign",
					_ => "unknown operation"
				},
				self.rows, self.cols, rhs.rows, rhs.cols
			);
		}
	}

	pub fn random(rows: usize, cols: usize) -> Self
	where Self: From<Vec<Vec<f64>>>
	{
		Self::from(
			(0..rows).into_iter().map(|_|
				(0..cols).into_iter().map(|_|
					thread_rng().gen::<f64>() * 2.0 - 1.0 // -1.0 to 1.0
				).collect()
			).collect::<Vec<Vec<f64>>>()
		)
	}

	pub fn hadamard(&self, rhs: &Self) -> Self
	where T: Mul, Vec<T>: FromIterator<T::Output>
	{
		self.confirm_dimension_operation(&rhs, "o");
		Self::from(
			self.data.iter().zip(rhs.data.iter()).map(|(u, v)|
				v.iter().zip(u.iter()).map(|(&a, &b)| a * b).collect()
			).collect::<Vec<Vec<T>>>()
		)
	}

	pub fn transpose(&self) -> Self {
		Self::from(
			(0..self.cols).into_iter().map(|j|
				(0..self.rows).into_iter().map(|i|
					self.data[i][j]
				).collect()
			).collect::<Vec<Vec<T>>>()
		)
	}
	
	pub fn map(&self, function: impl Fn(T) -> T) -> Self {
		Self::from(
			self.data.iter().map(|rows|
				rows.iter().map(|&value| function(value)).collect()
			).collect::<Vec<Vec<T>>>()
		)
	}
}

impl<T> From<Vec<Vec<T>>> for Matrix<T> {
	fn from(data: Vec<Vec<T>>) -> Self {
		Self {
			rows: data.len(),
			cols: data[0].len(),
			data
		}
	}
}

impl<T: Copy + Add> Add for Matrix<T>
where Vec<T>: FromIterator<T::Output>
{
	type Output = Self;
	fn add(self, rhs: Self) -> Self::Output {
		self.confirm_dimension_operation(&rhs, "+");
		Self::from(
			self.data.iter().zip(rhs.data.iter()).map(|(v, u)|
				v.iter().zip(u.iter()).map(|(&a, &b)| a + b).collect()
			).collect::<Vec<Vec<T>>>()
		)
	}
}

impl<T: Copy + Add> AddAssign for Matrix<T>
where Vec<T>: FromIterator<T::Output>
{
	fn add_assign(&mut self, rhs: Self) {
		self.confirm_dimension_operation(&rhs, "+=");
		self.data = self.data.iter().zip(rhs.data.iter()).map(|(v, u)|
			v.iter().zip(u.iter()).map(|(&a, &b)| a + b).collect()
		).collect()
	}
}

impl<T: Copy + Sub> Sub for Matrix<T>
where Vec<T>: FromIterator<T::Output>
{
	type Output = Self;
	fn sub(self, rhs: Self) -> Self::Output {
		self.confirm_dimension_operation(&rhs, "-");
		Self::from(
			self.data.iter().zip(rhs.data.iter()).map(|(v, u)|
				v.iter().zip(u.iter()).map(|(&a, &b)| a - b).collect()
			).collect::<Vec<Vec<T>>>()
		)
	}
}

impl<T: Copy + Mul> Mul for Matrix<T>
where Vec<T>: FromIterator<T::Output>,
T: Sum<T::Output>
{
	type Output = Self;
	fn mul(self, rhs: Self) -> Self::Output {
		self.confirm_dimension_operation(&rhs, "*");
		Self::from(
			self.data.iter().map(|v|
				rhs.transpose().data.iter().map(|u|
					v.iter().zip(u.iter()).map(|(&a, &b)| a * b).sum::<T>()
				).collect()
			).collect::<Vec<Vec<T>>>()
		)
	}
}

impl<T: Copy + Mul> MulAssign for Matrix<T>
where Vec<T>: FromIterator<T::Output>,
T: Sum<T::Output>
{
	fn mul_assign(&mut self, rhs: Self) {
		self.confirm_dimension_operation(&rhs, "*=");
		self.data = self.data.iter().map(|v|
			rhs.transpose().data.iter().map(|u|
				v.iter().zip(u.iter()).map(|(&a, &b)| a * b).sum::<T>()
			).collect()
		).collect::<Vec<Vec<T>>>();
		self.cols = rhs.cols;
	}
}

impl<T: ToString> Debug for Matrix<T> {
	fn fmt(&self, f: &mut Formatter) -> Result {
		write!(
			f,
			"{} x {} Matrix {{\n{}\n}}",
			self.rows,
			self.cols,
			(&self.data).into_iter()
			.map(|row| "  ".to_string()
				+ &row.into_iter()
				.map(|value| value.to_string())
				.collect::<Vec<String>>().join(" "))
			.collect::<Vec<String>>().join("\n")
		)
	}
}

#[cfg(test)]
mod tests {
	use super::Matrix as M;

	#[test]
	fn random_matrix() {
		let (m1, m2) = (
			M::random(3, 2),
			M::random(3, 2)
		);
		assert_ne!(m1, m2);
	}

	#[test]
	fn hadamard_product() {
		let (m1, m2) = (
			M::from(vec![
				vec![1.0, 2.0],
				vec![3.0, 4.0]]
			),
			M::from(vec![
				vec![1.0, 3.0],
				vec![5.0, 7.0]]
			)
		);
		assert_eq!(
			M::hadamard(&m2, &m1),
			M::from(vec![
				vec![1.0, 6.0],
				vec![15.0, 28.0],
			])
		);
	}

	#[test]
	fn transpose_matrices() {
		let m1 = M::from(vec![vec![1.0; 3]; 2]);
		let m2 = M::from(vec![vec![1.0; 2]; 3]);
		assert_eq!(m1.transpose(), m2);
		assert_eq!(m2.transpose(), m1);
	}

	#[test]
	fn map_matrix() {
		let m1 = M::from(vec![vec![3.0; 3]])
			.map(|x| x * 2.0);
		assert_eq!(m1, M::from(vec![vec![6.0; 3]]));
	}

	#[test]
	fn from_matrix() {
		let (m1, m2) = (
			M::from(vec![vec![1.0; 3]; 2]),
			M { rows: 2, cols: 3, data: vec![vec![1.0; 3]; 2]}
		);
		assert_eq!(m1, m2);
	}

	#[test]
	fn add_matrices() {
		let (m1, m2) = (
			M::from(vec![vec![1.0; 3]; 2]),
			M::from(vec![
				vec![1.0, 3.0, 5.0],
				vec![7.0, 9.0, 11.0]]
			)
		);
		assert_eq!(m1 + m2,
			M::from(vec![
				vec![2.0, 4.0, 6.0],
				vec![8.0, 10.0, 12.0]]
			)
		);
	}

	#[test]
	fn add_and_assign() {
		let mut m1 = M::from(vec![vec![1.0; 3]; 2]);
		m1 += M::from(vec![vec![2.0; 3]; 2]);
		assert_eq!(m1, M::from(vec![vec![3.0; 3]; 2]));
	}

	#[test]
	fn subtract_matrices() {
		let (m1, m2) = (
			M::from(vec![vec![1.0; 3]; 2]),
			M::from(vec![
				vec![1.0, 3.0, 5.0],
				vec![7.0, 9.0, 11.0]]
			)
		);
		assert_eq!(m2 - m1,
			M::from(vec![
				vec![0.0, 2.0, 4.0],
				vec![6.0, 8.0, 10.0]]
			)
		);
	}

	#[test]
	fn dot_product() {
		let (m1, m2) = (
			M::from(vec![
				vec![1.0; 2],
				vec![2.0; 2],
				vec![3.0; 2],
				vec![4.0; 2]]
			),
			M::from(vec![
				vec![1.0, 3.0, 5.0],
				vec![7.0, 9.0, 11.0]]
			)
		);
		assert_eq!(m1 * m2,
			M::from(vec![
				vec![8.0, 12.0, 16.0],
				vec![16.0, 24.0, 32.0],
				vec![24.0, 36.0, 48.0],
				vec![32.0, 48.0, 64.0]]
			)
		);
	}

	#[test]
	fn multiply_and_assign() {
		let mut m1 = M::from(vec![vec![1.0; 2]]);
		m1 *= M::from(vec![vec![1.0, 3.0, 5.0], vec![7.0, 9.0, 11.0]]);
		assert_eq!(m1, M::from(vec![vec![8.0, 12.0, 16.0]]));
	}

	#[test]
	#[should_panic]
	fn multiplication_panic() {
		let m1 = M::from(vec![vec![1.0; 2]; 4]);
		let m2 = M::from(vec![vec![1.0, 3.0]]);
		let _ = m2 * m1;
	}

	#[test]
	fn order_of_assignment_operations() {
		let mut m1 = M::from(vec![vec![2.0; 3]; 3]);
		let m2 =  M::from(vec![vec![3.0; 3]; 3]);
		let m3 = M::from(vec![vec![4.0; 3]; 3]);
		m1 *= m2.clone() + m3.clone(); // m1 = m1 * (m2 + m3):
		m1 += m2.clone() * m3.clone(); // m1 = m1 + (m2 * m3):
		assert_eq!(m1, M::from(vec![vec![78.0; 3]; 3]));
	}
}