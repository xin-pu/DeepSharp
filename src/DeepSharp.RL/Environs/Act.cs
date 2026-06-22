namespace DeepSharp.RL.Environs
{
	/// <summary>
	///     Action
	/// </summary>
	public class Act : IEqualityComparer<Act>
	{
		public Act(torch.Tensor? action)
		{
			Value     = action;
			TimeStamp = DateTime.Now;
		}

		/// <summary>
		///     Tensor format of the action.
		/// </summary>
		public torch.Tensor? Value { get; set; }

		/// <summary>
		///     Timestamp when the action was generated.
		/// </summary>
		public DateTime TimeStamp { get; set; }


		public bool Equals(Act? x, Act? y)
		{
			if (ReferenceEquals(x, y)) return true;
			if (ReferenceEquals(x, null)) return false;
			if (ReferenceEquals(y, null)) return false;
			return x.GetType() == y.GetType() && x.Value!.Equals(y.Value!);
		}

		public int GetHashCode(Act obj)
		{
			return HashCode.Combine(obj.TimeStamp, obj.Value);
		}

		public Act To(torch.Device device)
		{
			return new Act(Value!.to(device));
		}

		public override string ToString()
		{
			return $"{TimeStamp}\t{Value!.ToString(torch.numpy)}";
		}
	}
}