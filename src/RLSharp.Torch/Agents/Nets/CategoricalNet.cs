namespace RLSharp.Torch.Agents
{
	/// <summary>
	///     Categorical Network (C51).
	///     Outputs action value distribution logits, shape [batch, actionSize, numAtoms].
	///     Each action has a probability distribution over numAtoms support points.
	/// </summary>
	public sealed class CategoricalNet : Module<torch.Tensor, torch.Tensor>
	{
		private readonly long                               _actionSize;
		private readonly Module<torch.Tensor, torch.Tensor> _layers;
		private readonly long                               _numAtoms;

		public CategoricalNet(long obsSize,
			long                   hiddenSize,
			long                   actionSize,
			long                   numAtoms   = 51,
			DeviceType             deviceType = DeviceType.CPU)
			: base("CategoricalNet")
		{
			_actionSize = actionSize;
			_numAtoms   = numAtoms;

			_layers = Sequential(
				("fc1", Linear(obsSize, hiddenSize)),
				("relu1", ReLU()),
				("fc2", Linear(hiddenSize, hiddenSize)),
				("relu2", ReLU()),
				("fc3", Linear(hiddenSize, actionSize * numAtoms))
			);

			_layers.to(new torch.Device(deviceType));
			RegisterComponents();
		}

		public override torch.Tensor forward(torch.Tensor input)
		{
			// input: [batch, obsSize] â†?output: [batch, actionSize, numAtoms]
			var x      = _layers.forward(input.to_type(torch.ScalarType.Float32));
			var logits = x.view(-1, _actionSize, _numAtoms);
			// Return logits (softmax applied by caller)
			return logits;
		}

		protected override void Dispose(bool disposing)
		{
			if (disposing)
			{
				_layers.Dispose();
				ClearModules();
			}

			base.Dispose(disposing);
		}
	}
}