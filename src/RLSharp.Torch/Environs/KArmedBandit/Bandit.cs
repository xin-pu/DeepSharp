namespace RLSharp.Torch.Environs
{
	/// <summary>
	///     Simplified bandit that outputs a coin with probability Prob.
	/// </summary>
	public class Bandit
	{
		public Bandit(string name, double prob = 0.7)
		{
			Name         = name;
			Prob         = prob;
			RandomSource = new Random();
		}

		protected Random RandomSource { get; set; }

		public double Prob { get; set; }

		public string Name { get; set; }


		public float Step()
		{
			var pro = RandomSource.NextDouble();
			return pro <= Prob ? 1 : 0;
		}

		public override string ToString()
		{
			return $"Bandit{Name}:\t{Prob:P}";
		}
	}
}