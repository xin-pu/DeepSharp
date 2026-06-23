namespace DeepSharp.RL.Agents
{
	/// <summary>
	///     Noisy Network.
	///     Uses NoisyLinear instead of standard Linear for parameterized exploration.
	///     Call ResetNoise() to sample new noise; the network has built-in exploration.
	/// </summary>
	public sealed class NoisyNet : Module<torch.Tensor, torch.Tensor>
	{
		private readonly NoisyLinear _fc1;
		private readonly NoisyLinear _fc2;

		public NoisyNet(long obsSize, long hiddenSize, long actionNum, DeviceType deviceType = DeviceType.CPU)
			: base("NoisyNet")
		{
			_fc1 = new NoisyLinear(obsSize, hiddenSize);
			_fc2 = new NoisyLinear(hiddenSize, actionNum);

			_fc1.to(new torch.Device(deviceType));
			_fc2.to(new torch.Device(deviceType));

			RegisterComponents();
		}

		/// <summary>
		///     Reset noise for all NoisyLinear layers.
		/// </summary>
		public void ResetNoise()
		{
			_fc1.ResetNoise();
			_fc2.ResetNoise();
		}

		public override torch.Tensor forward(torch.Tensor input)
		{
			var x = _fc1.forward(input.to_type(torch.ScalarType.Float32));
			x = functional.relu(x);
			return _fc2.forward(x);
		}

		protected override void Dispose(bool disposing)
		{
			if (disposing)
			{
				_fc1.Dispose();
				_fc2.Dispose();
				ClearModules();
			}

			base.Dispose(disposing);
		}
	}
}