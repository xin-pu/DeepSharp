namespace DeepSharp.RL.Environs
{
	/// <summary>
	///     Observation
	/// </summary>
	public class Observation
	{
		public Observation(torch.Tensor? state)
		{
			Value     = state;
			TimeStamp = DateTime.Now;
		}

		/// <summary>
		///     Tensor format of the observation.
		/// </summary>
		public torch.Tensor? Value { get; set; }

		/// <summary>
		///     Timestamp when the observation was generated.
		/// </summary>
		public DateTime TimeStamp { get; set; }

		public Observation To(torch.Device device)
		{
			return new Observation(Value?.to(device));
		}


		public object Clone()
		{
			return new Observation(Value) { TimeStamp = TimeStamp };
		}

		public override string ToString()
		{
			return $"Observation\r\n{Value?.ToString(torch.numpy)}";
		}
	}
}