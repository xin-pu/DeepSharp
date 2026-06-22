namespace DeepSharp.RL.ActionSelectors
{
	/// <summary>
	///     Action Selector helps convert network predictions into specific action objects.
	/// </summary>
	public abstract class ActionSelector
	{
		protected ActionSelector(bool keepDims = false)
		{
			KeepDims = keepDims;
		}

		public bool KeepDims { get; set; }

		public abstract torch.Tensor Select(torch.Tensor probs);
	}
}