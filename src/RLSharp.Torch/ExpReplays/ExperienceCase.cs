namespace RLSharp.Torch.ExperienceSources
{
	/// <summary>
	/// </summary>
	public sealed class ExperienceCase : IDisposable
	{
		public ExperienceCase(torch.Tensor preState,
			torch.Tensor                   action,
			torch.Tensor                   reward,
			torch.Tensor                   postState,
			torch.Tensor                   done)
		{
			PreState  = preState;
			Action    = action;
			Reward    = reward;
			PostState = postState;
			Done      = done;
		}

		/// <summary>
		///     State before action.
		/// </summary>
		public torch.Tensor PreState { get; set; }

		/// <summary>
		///     Action taken.
		/// </summary>
		public torch.Tensor Action { get; set; }

		/// <summary>
		///     Reward received.
		/// </summary>
		public torch.Tensor Reward { get; set; }

		/// <summary>
		///     State after action.
		/// </summary>
		public torch.Tensor PostState { get; set; }

		/// <summary>
		///     Whether the episode is complete.
		/// </summary>
		public torch.Tensor Done { get; set; }

		public void Dispose()
		{
			PreState.Dispose();
			Action.Dispose();
			Reward.Dispose();
			PostState.Dispose();
			Done.Dispose();
		}
	}
}