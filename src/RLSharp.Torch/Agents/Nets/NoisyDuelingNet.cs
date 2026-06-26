namespace RLSharp.Torch.Agents
{
	/// <summary>
	///     Noisy Dueling Network.
	///     Combines NoisyLinear noise exploration with Dueling architecture:
	///     - Uses NoisyLinear instead of standard Linear for built-in parameterized exploration
	///     - Dueling structure: Q(s,a) = V(s) + A(s,a) - mean(A)
	/// </summary>
	public sealed class NoisyDuelingNet : Module<torch.Tensor, torch.Tensor>
	{
		private readonly Module<torch.Tensor, torch.Tensor> _advantageOut;
		private readonly NoisyLinear                        _fcAdvantage;
		private readonly NoisyLinear                        _fcFeature;
		private readonly NoisyLinear                        _fcValue;

		private readonly Module<torch.Tensor, torch.Tensor> _valueOut;

		public NoisyDuelingNet(long obsSize,
			long                    hiddenSize,
			long                    actionNum,
			DeviceType              deviceType = DeviceType.CPU)
			: base("NoisyDuelingNet")
		{
			_fcFeature   = new NoisyLinear(obsSize, hiddenSize);
			_fcValue     = new NoisyLinear(hiddenSize, hiddenSize / 2);
			_fcAdvantage = new NoisyLinear(hiddenSize, hiddenSize / 2);

			_fcFeature.to(new torch.Device(deviceType));
			_fcValue.to(new torch.Device(deviceType));
			_fcAdvantage.to(new torch.Device(deviceType));

			RegisterComponents();

			// Value and Advantage output heads use standard Linear (no noise needed)
			_valueOut     = Linear(hiddenSize / 2, 1);
			_advantageOut = Linear(hiddenSize / 2, actionNum);

			_valueOut.to(new torch.Device(deviceType));
			_advantageOut.to(new torch.Device(deviceType));
		}

		/// <summary>
		///     Reset noise for all NoisyLinear layers.
		/// </summary>
		public void ResetNoise()
		{
			_fcFeature.ResetNoise();
			_fcValue.ResetNoise();
			_fcAdvantage.ResetNoise();
		}

		public override torch.Tensor forward(torch.Tensor input)
		{
			var x = _fcFeature.forward(input.to_type(torch.ScalarType.Float32));
			x = functional.relu(x);

			var v = functional.relu(_fcValue.forward(x));
			v = _valueOut.forward(v); // [batch, 1]

			var a = functional.relu(_fcAdvantage.forward(x));
			a = _advantageOut.forward(a); // [batch, actionSize]

			// Q(s,a) = V(s) + A(s,a) - mean(A)
			return v + (a - a.mean(new long[] { 1 }, true));
		}

		protected override void Dispose(bool disposing)
		{
			if (disposing)
			{
				_fcFeature.Dispose();
				_fcValue.Dispose();
				_fcAdvantage.Dispose();
				_valueOut.Dispose();
				_advantageOut.Dispose();
				ClearModules();
			}

			base.Dispose(disposing);
		}
	}
}