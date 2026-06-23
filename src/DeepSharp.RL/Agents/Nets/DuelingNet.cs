namespace DeepSharp.RL.Agents
{
	/// <summary>
	///     Dueling Network architecture.
	///     Decomposes Q(s,a) into state value V(s) and advantage A(s,a):
	///     Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
	///     Shared feature extractor → V head (output 1) + A head (output actionSize)
	/// </summary>
	public sealed class DuelingNet : Module<torch.Tensor, torch.Tensor>
	{
		private readonly Module<torch.Tensor, torch.Tensor> _advantage;
		private readonly Module<torch.Tensor, torch.Tensor> _feature;
		private readonly Module<torch.Tensor, torch.Tensor> _value;

		public DuelingNet(long obsSize, long hiddenSize, long actionNum, DeviceType deviceType = DeviceType.CPU)
			: base("DuelingNet")
		{
			// Shared feature extractor
			_feature = Sequential(
				("fc1", Linear(obsSize, hiddenSize)),
				("relu", ReLU())
			);

			// V head: state value
			_value = Sequential(
				("fc_v", Linear(hiddenSize, hiddenSize / 2)),
				("relu_v", ReLU()),
				("out_v", Linear(hiddenSize / 2, 1))
			);

			// A head: action advantage
			_advantage = Sequential(
				("fc_a", Linear(hiddenSize, hiddenSize / 2)),
				("relu_a", ReLU()),
				("out_a", Linear(hiddenSize / 2, actionNum))
			);

			_feature.to(new torch.Device(deviceType));
			_value.to(new torch.Device(deviceType));
			_advantage.to(new torch.Device(deviceType));

			RegisterComponents();
		}

		public override torch.Tensor forward(torch.Tensor input)
		{
			var x = _feature.forward(input.to_type(torch.ScalarType.Float32));
			var v = _value.forward(x);     // [batch, 1]
			var a = _advantage.forward(x); // [batch, actionSize]
			// Q(s,a) = V(s) + A(s,a) - mean(A)
			return v + (a - a.mean(new long[] { 1 }, true));
		}

		protected override void Dispose(bool disposing)
		{
			if (disposing)
			{
				_feature.Dispose();
				_value.Dispose();
				_advantage.Dispose();
				ClearModules();
			}

			base.Dispose(disposing);
		}
	}
}