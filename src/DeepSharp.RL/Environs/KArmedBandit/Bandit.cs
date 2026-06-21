namespace DeepSharp.RL.Environs
{
	/// <summary>
	///     简化的赌博机，以Prob 概率吐出一枚硬币
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