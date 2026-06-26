namespace RLSharp.Torch.Environs
{
	/// <summary>
	///     Action
	/// </summary>
	public class ActionValue : IEqualityComparer<ActionValue>
	{
		public ActionValue(torch.Tensor? action)
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


		public bool Equals(ActionValue? x, ActionValue? y)
		{
			if (ReferenceEquals(x, y)) return true;
			if (ReferenceEquals(x, null)) return false;
			if (ReferenceEquals(y, null)) return false;
			return x.GetType() == y.GetType() && x.Value!.Equals(y.Value!);
		}

		public int GetHashCode(ActionValue obj)
		{
			return HashCode.Combine(obj.TimeStamp, obj.Value);
		}

		public ActionValue To(torch.Device device)
		{
			return new ActionValue(Value!.to(device));
		}

		public override string ToString()
		{
			return $"{TimeStamp}\t{Value!.ToString(torch.numpy)}";
		}
	}
}