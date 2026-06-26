using TorchSharp.Modules;

namespace RLSharp.Torch.Agents
{
	/// <summary>
	///     Noisy Linear Layer (Factorized Gaussian Noise).
	///     Uses learnable noise parameters instead of ε-greedy exploration:
	///     y = (μ_w + σ_w �?ε_w) @ x + (μ_b + σ_b �?ε_b)
	///     Noise ε uses factorized Gaussian to reduce parameter count:
	///     ε_w = f(ε_i) * f(ε_j)^T
	///     f(x) = sign(x) * sqrt(|x|)
	/// </summary>
	public class NoisyLinear : Module<torch.Tensor, torch.Tensor>
	{
		private readonly long         _inFeatures;
		private readonly long         _outFeatures;
		private          torch.Tensor _epsilonBias = null!;

		private torch.Tensor _epsilonWeight = null!;

		/// <param name="inFeatures">Input feature dimension.</param>
		/// <param name="outFeatures">Output feature dimension.</param>
		/// <param name="sigmaInit">Initial standard deviation σ (default 0.017, from NoisyNet paper).</param>
		public NoisyLinear(long inFeatures, long outFeatures, float sigmaInit = 0.017f)
			: base("NoisyLinear")
		{
			_inFeatures  = inFeatures;
			_outFeatures = outFeatures;

			var muRange = 1.0f / MathF.Sqrt(inFeatures);

			// μ weight: uniform distribution [-muRange, muRange]
			MuWeight = new Parameter(torch.empty(outFeatures, inFeatures).uniform_(-muRange, muRange));
			// σ weight: constant sigmaInit / sqrt(inFeatures)
			SigmaWeight =
				new Parameter(torch.full(new[] { outFeatures, inFeatures }, sigmaInit / MathF.Sqrt(inFeatures)));

			// μ bias
			MuBias = new Parameter(torch.empty(outFeatures).uniform_(-muRange, muRange));
			// σ bias
			SigmaBias = new Parameter(torch.full(new[] { outFeatures }, sigmaInit / MathF.Sqrt(inFeatures)));

			RegisterComponents();
			ResetNoise();
		}

		public Parameter MuWeight { get; }

		public Parameter SigmaWeight { get; }

		public Parameter MuBias { get; }

		public Parameter SigmaBias { get; }

		/// <summary>
		///     Re-sample noise (call before each episode or at the start of Learn).
		/// </summary>
		public void ResetNoise()
		{
			// Factorized Gaussian noise:
			// ε_w = f(ε_i) * f(ε_j)^T
			// f(x) = sign(x) * sqrt(|x|)

			var epsIn  = FactorisedNoise(_inFeatures);
			var epsOut = FactorisedNoise(_outFeatures);

			_epsilonWeight = epsOut.outer(epsIn);
			_epsilonBias   = epsOut;
		}

		private static torch.Tensor FactorisedNoise(long size)
		{
			var noise = torch.randn(size);
			return noise.sign() * noise.abs().sqrt();
		}

		public override torch.Tensor forward(torch.Tensor input)
		{
			// y = (μ_w + σ_w �?ε_w) @ x + (μ_b + σ_b �?ε_b)
			var weight = MuWeight + SigmaWeight * _epsilonWeight;
			var bias   = MuBias   + SigmaBias   * _epsilonBias;
			return functional.linear(input, weight, bias);
		}

		protected override void Dispose(bool disposing)
		{
			if (disposing)
			{
				_epsilonWeight?.Dispose();
				_epsilonBias?.Dispose();
				MuWeight.Dispose();
				SigmaWeight.Dispose();
				MuBias.Dispose();
				SigmaBias.Dispose();
				ClearModules();
			}

			base.Dispose(disposing);
		}
	}
}