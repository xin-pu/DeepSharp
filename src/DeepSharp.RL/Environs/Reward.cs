namespace DeepSharp.RL.Environs
{
	/// <summary>
	///     Reward
	/// </summary>
	public class Reward
	{
		public Reward(float value)
		{
			Value     = value;
			TimeStamp = DateTime.Now;
		}

		/// <summary>
		///     reward
		/// </summary>
		public float Value { get; set; }

		/// <summary>
		///     TimeStamp of get reward
		/// </summary>
		public DateTime TimeStamp { get; set; }

		public override string ToString()
		{
			return $"Reward:\t{Value}";
		}
	}
}