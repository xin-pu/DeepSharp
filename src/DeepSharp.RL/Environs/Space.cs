using System.Diagnostics;

namespace DeepSharp.RL.Environs
{
	/// <summary>
	///     Base class for both action space and observation space.
	/// </summary>
	public abstract class Space : IDisposable
	{
		protected Space(
			long[]           shape,
			torch.ScalarType type,
			DeviceType       deviceType,
			long             seed)
		{
			(Shape, Type, DeviceType) = (shape, type, deviceType);
			CheckInitParameter(shape, type);
			CheckType();
			Generator = torch.random.manual_seed(seed);
			N         = shape.Aggregate(1, (a, b) => (int)(a * b));
		}

		public long N { get; init; }

		public long[] Shape { get; }

		public torch.ScalarType Type { get; }

		public DeviceType DeviceType { get; }

		internal torch.Generator Generator { get; }

		internal torch.Device Device => new(DeviceType);

		public void Dispose()
		{
			Generator.Dispose();
		}

		/// <summary>
		///     Returns a randomly sampled tensor from the space.
		/// </summary>
		/// <returns></returns>
		public abstract torch.Tensor Sample();

		public abstract void CheckType();

		/// <summary>
		///     Generates a zero tensor consistent with the space shape and type.
		/// </summary>
		/// <returns></returns>
		public virtual torch.Tensor Generate()
		{
			return torch.zeros(Shape, Type, Device);
		}


		public override string ToString()
		{
			return $"Space Type: {GetType().Name}\nShape: {Shape}\ndType: {Type} \nN:{N}";
		}

		private static void CheckInitParameter(long[] shape, torch.ScalarType type)
		{
			Debug.Assert(shape        != null);
			Debug.Assert(shape.Length > 0);
		}
	}
}