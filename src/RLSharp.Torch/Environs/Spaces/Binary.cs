namespace RLSharp.Torch.Environs.Spaces
{
	/// <summary>
	///     It's Space only support [0,1]
	/// </summary>
	public class Binary : Discrete
	{
		public Binary(torch.ScalarType dtype      = torch.ScalarType.Int32,
			DeviceType                 deviceType = DeviceType.CPU,
			long                       seed       = 1)
			: base(2, dtype, deviceType, seed)
		{
			N = 1;
		}
	}
}