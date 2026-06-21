namespace TorchSharpTest
{
	public class AbstractTest
	{
		private readonly ITestOutputHelper _testOutputHelper;

		public AbstractTest(ITestOutputHelper testOutputHelper)
		{
			_testOutputHelper = testOutputHelper;
		}

		protected void writeLine(object? obj)
		{
			_testOutputHelper.WriteLine(obj?.ToString());
		}

		internal void Print(string[] objs)
		{
			foreach (var o in objs) Print(o);
		}

		internal void Print(object obj)
		{
			writeLine(obj);
		}


		/// <summary>
		///     Todo optimize print Tensor
		/// </summary>
		/// <param name="tensor"></param>
		internal void Print(Tensor tensor)
		{
			writeLine(tensor.ToString(numpy));
			writeLine(tensor);
			writeLine("");
		}

		/// <summary>
		///     Todo optimize print Tensor
		/// </summary>
		/// <param name="tensor"></param>
		internal void Print(float tensor)
		{
			writeLine(tensor);
		}
	}
}