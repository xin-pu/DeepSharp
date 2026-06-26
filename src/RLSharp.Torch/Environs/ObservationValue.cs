namespace RLSharp.Torch.Environs
{
	/// <summary>
	///     ObservationValue
	/// </summary>
	public class ObservationValue
	{
		public ObservationValue(torch.Tensor? state)
		{
			Value     = state;
			TimeStamp = DateTime.Now;
		}

		/// <summary>
		///     Tensor format of the ObservationValue.
		/// </summary>
		public torch.Tensor? Value { get; set; }

		/// <summary>
		///     Timestamp when the ObservationValue was generated.
		/// </summary>
		public DateTime TimeStamp { get; set; }

		public ObservationValue To(torch.Device device)
		{
			return new ObservationValue(Value?.to(device));
		}


		public object Clone()
		{
			return new ObservationValue(Value) { TimeStamp = TimeStamp };
		}

		public override string ToString()
		{
			return $"ObservationValue\r\n{Value?.ToString(torch.numpy)}";
		}
	}
}