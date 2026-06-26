namespace RLSharp.Torch.Environs
{
	/// <summary>
	///     Reward value with timestamp.
	/// </summary>
	public class Reward
	{
		public Reward(float value)
		{
			Value     = value;
			TimeStamp = DateTime.Now;
		}

		/// <summary>
		///     The reward value.
		/// </summary>
		public float Value { get; set; }

		/// <summary>
		///     Timestamp when the reward was received.
		/// </summary>
		public DateTime TimeStamp { get; set; }

		public override string ToString()
		{
			return $"Reward:\t{Value}";
		}
	}
}