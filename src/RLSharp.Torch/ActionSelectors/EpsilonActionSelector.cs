namespace RLSharp.Torch.ActionSelectors
{
	public class EpsilonActionSelector : ActionSelector
	{
		public EpsilonActionSelector(ActionSelector selector, float epsilon = 0.1f)
		{
			ArgumentNullException.ThrowIfNull(selector);
			if (epsilon is < 0f or > 1f)
				throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be between 0 and 1.");

			Selector = selector;
			Epsilon  = epsilon;
		}

		public ActionSelector Selector { get; protected set; }

		public float Epsilon { get; set; }

		public override torch.Tensor Select(torch.Tensor probs)
		{
			if (probs.dim() is not (1 or 2))
				throw new NotSupportedException("Only one- and two-dimensional tensors are supported.");

			using var greedy      = Selector.Select(probs);
			var       actionCount = probs.shape[^1];
			var       batchSize   = probs.dim() == 1 ? 1 : probs.shape[0];
			using var randomActions = torch.randint(
				0,
				actionCount,
				new[] { batchSize },
				torch.ScalarType.Int64,
				probs.device);
			using var randomValues = torch.rand(new[] { batchSize }, device: probs.device);
			using var explore      = randomValues < Epsilon;
			var       selected     = torch.where(explore, randomActions, greedy.reshape(batchSize));

			if (probs.dim() == 1)
				return selected.squeeze(0);

			return KeepDims ? selected.unsqueeze(-1) : selected;
		}
	}
}