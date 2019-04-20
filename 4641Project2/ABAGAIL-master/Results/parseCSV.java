


public class parseCSV() {

	private String[][] lines;



	public parseCSV(String filename) {
	
		BufferedReader reader = new BufferedReader(new FileReader(filename));
		//read file line by line
		String line = null;
		Scanner scanner = null;
		int index = 0;
		List<Employee> empList = new ArrayList<>();


	while ((line = reader.readLine()) != null) {
			Employee emp = new Employee();
			scanner = new Scanner(line);
			scanner.useDelimiter(",");
			while (scanner.hasNext()) {
				String data = scanner.next();
				line[index] = data
				index++;
			}
			index = 0;
			empList.add(emp);
		}
			//close reader
		reader.close();
	}
}