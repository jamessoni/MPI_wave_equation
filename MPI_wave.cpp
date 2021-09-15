#define _USE_MATH_DEFINES

#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <time.h>
#include <cmath>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>  
#include <cstdlib> 

using namespace std;

int id, p;
int tag_num = 1;

double y_max = 10.0, x_max = 10.0, dx, dy;
double t, t_out = 0.0, dt_out = 0.04, dt; //having dt_out at 0.04 -> equivalent to computing 25 frames a second output timestep
double c = 1;
double t_max = 30.0;

int m, n; // rows and columns of processes
int imax, jmax; // rows and columns of problem's domain
int id_row, id_col; // position of process subdomain in overall grid
int local_rows, local_cols;  // rows and columns of subdomain

// uncomment/select boundary condition choice
string boundary = "periodic";
//string boundary = "dirichlet";
//string boundary = "neumann";

class mpi_class
{
	// The purpose of this class is to create two vectors: send datatypes, receive datatypes

public:
	int local_rows;
	int local_cols;
	double** grid;
	vector<MPI_Datatype> typelist_send_centre; // to be filled with 4 send datatypes
	vector<MPI_Datatype> typelist_recv_centre; // to be filled with 4 recv datatypes

	// constructor (including padding rows and cols)
	mpi_class(int& local_rows, int& local_cols, double**& grid)
	{
		this->local_rows = local_rows;
		this->local_cols = local_cols;
		this->grid = new double* [this->local_rows + 2];
		for (int i = 0; i < local_rows + 2; i++)
		{
			this->grid[i] = new double[local_cols + 2];
			for (int j = 0; j < local_cols + 2; j++)
			{
				this->grid[i][j] = grid[i][j];
			}
		}
	}
	// init datatypes
	MPI_Datatype mpi_send_left, mpi_send_right, mpi_send_top, mpi_send_bottom;
	MPI_Datatype mpi_recv_right, mpi_recv_left, mpi_recv_top, mpi_recv_bottom;
	void mpi_sendtype();
	void mpi_recvtype();
};

void mpi_class::mpi_sendtype()
{
	// creating 4 send datatypes -> right, left, top, bottom
	vector<int> block_length;
	vector<MPI_Aint> addresses;
	vector<MPI_Datatype> typelist;

	// left side send - going down the rows
	for (int i = 1; i <= this->local_rows; i++)
	{
		block_length.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Aint temp_add;
		MPI_Get_address(&grid[i][1], &temp_add);
		addresses.push_back(temp_add);
	}
	// not requiring offsets as using MPI_Bottom as the pointer
	MPI_Type_create_struct(block_length.size(), &block_length[0], &addresses[0], &typelist[0], &this->mpi_send_left);
	MPI_Type_commit(&this->mpi_send_left);
	this->typelist_send_centre.push_back(this->mpi_send_left);
	block_length.clear();
	addresses.clear();
	typelist.clear();

	// right side send - going down the rows
	// Repitition of above
	for (int i = 1; i <= this->local_rows; i++)
	{
		block_length.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Aint temp_add;
		MPI_Get_address(&grid[i][local_cols], &temp_add);
		addresses.push_back(temp_add);
	}

	MPI_Type_create_struct(block_length.size(), &block_length[0], &addresses[0], &typelist[0], &this->mpi_send_right);
	MPI_Type_commit(&this->mpi_send_right);
	this->typelist_send_centre.push_back(this->mpi_send_right);
	block_length.clear();
	addresses.clear();
	typelist.clear();

	// top side send - 
	// Repitition of above
	for (int j = 1; j <= this->local_cols; j++)
	{
		block_length.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Aint temp_add;
		MPI_Get_address(&grid[1][j], &temp_add);
		addresses.push_back(temp_add);
	}

	MPI_Type_create_struct(block_length.size(), &block_length[0], &addresses[0], &typelist[0], &this->mpi_send_top);
	MPI_Type_commit(&this->mpi_send_top);
	this->typelist_send_centre.push_back(this->mpi_send_top);
	block_length.clear();
	addresses.clear();
	typelist.clear();

	// bottom side send - 
	// Repition of above
	for (int j = 1; j <= this->local_cols; j++)
	{
		block_length.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Aint temp_add;
		MPI_Get_address(&grid[local_rows][j], &temp_add);
		addresses.push_back(temp_add);
	}
	// 
	MPI_Type_create_struct(block_length.size(), &block_length[0], &addresses[0], &typelist[0], &this->mpi_send_bottom);
	MPI_Type_commit(&this->mpi_send_bottom);
	this->typelist_send_centre.push_back(this->mpi_send_bottom);
	block_length.clear();
	addresses.clear();
	typelist.clear();

}

void mpi_class::mpi_recvtype()
{
	// creating 4 receive datatypes -> right, left, top, bottom
	// The order in which these datatypes are pushed into the vector is opposite
	// to the order of datatypes in the send vector - so that they match
	// e.g. left col is matched with right, top with bottom

	vector<int> block_length;
	vector<MPI_Aint> addresses;
	vector<MPI_Datatype> typelist;

	// right side receive - going down the rows
	for (int i = 1; i <= this->local_rows; i++)
	{
		block_length.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Aint temp_add;
		MPI_Get_address(&grid[i][local_cols + 1], &temp_add);
		addresses.push_back(temp_add);
	}
	// Again not using offsets, using MPI_BOTTOM
	MPI_Type_create_struct(block_length.size(), &block_length[0], &addresses[0], &typelist[0], &this->mpi_recv_right);
	MPI_Type_commit(&this->mpi_recv_right);
	this->typelist_recv_centre.push_back(this->mpi_recv_right);
	block_length.clear();
	addresses.clear();
	typelist.clear();

	// left side receive - going down the rows
	for (int i = 1; i <= this->local_rows; i++)
	{
		block_length.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Aint temp_add;
		MPI_Get_address(&grid[i][0], &temp_add);
		addresses.push_back(temp_add);
	}

	MPI_Type_create_struct(block_length.size(), &block_length[0], &addresses[0], &typelist[0], &this->mpi_recv_left);
	MPI_Type_commit(&this->mpi_recv_left);
	this->typelist_recv_centre.push_back(this->mpi_recv_left);
	block_length.clear();
	addresses.clear();
	typelist.clear();

	// bottom side receive - going across columns
	for (int j = 1; j <= this->local_cols; j++)
	{
		block_length.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Aint temp_add;
		MPI_Get_address(&grid[local_rows + 1][j], &temp_add);
		addresses.push_back(temp_add);
	}

	MPI_Type_create_struct(block_length.size(), &block_length[0], &addresses[0], &typelist[0], &this->mpi_recv_bottom);
	MPI_Type_commit(&this->mpi_recv_bottom);
	this->typelist_recv_centre.push_back(this->mpi_recv_bottom);
	block_length.clear();
	addresses.clear();
	typelist.clear();

	// top side receive - going across columns
	for (int j = 1; j <= this->local_cols; j++)
	{
		block_length.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Aint temp_add;
		MPI_Get_address(&grid[0][j], &temp_add);
		addresses.push_back(temp_add);
	}

	MPI_Type_create_struct(block_length.size(), &block_length[0], &addresses[0], &typelist[0], &this->mpi_recv_top);
	MPI_Type_commit(&this->mpi_recv_top);
	this->typelist_recv_centre.push_back(this->mpi_recv_top);
	block_length.clear();
	addresses.clear();
	typelist.clear();
}

// setting up continuous 1d and 2d arrays
// used by all procs 
// sets up main grids -> grid, old, new
// subdomain grids, padded grids
void setup_continuous_array(double**& array_2d, double*& array_1d, int local_rows, int local_cols)
{
	array_1d = new double[(local_rows) * (local_cols)];
	for (int i = 0; i < (local_rows) * (local_cols); i++) {
		array_1d[i] = 0;
	}
	array_2d = new double* [local_rows];

	// Initialising the 2_d array
	for (int i = 0; i < local_rows; i++) {
		array_2d[i] = new double[local_cols];
		for (int j = 0; j < local_cols; j++) {
			array_2d[i][j] = array_1d[i * (local_cols)+j];
		}
	}
}
// called in deletion routines
void free_2d_array(double**& array_2d, int rows)
{
	for (int j = 0; j < rows; j++) {
		delete[] array_2d[j];
	}
	delete[] array_2d;
}

// pads the grid and allocates its interior/central region with sub-domain grid values
// also allocates the padded 2d grid
void padding_grid(double*& grid_1d, double*& grid_padded_1d, double**& grid_padded_2d, int local_rows, int local_cols)
{
	for (int i = 1; i <= local_rows; i++) {
		for (int j = 1; j <= local_cols; j++) {
			grid_padded_1d[i * (local_cols + 2) + j] = grid_1d[(i - 1) * local_cols + j - 1];
		}
	}

	for (int i = 0; i < local_rows + 2; i++) {
		for (int j = 0; j < local_cols + 2; j++) {
			grid_padded_2d[i][j] = grid_padded_1d[i * (local_cols + 2) + j];
			grid_padded_2d[i][j] = grid_padded_1d[i * (local_cols + 2) + j];
		}
	}
}

//sets half sinusoidal intitial disturbance - this is brute force
// - it can be done more elegantly
void initial_disturbance(double*& grid_1d, double*& old_grid_1d, int imax, int jmax)
{
	double r_splash = 1.0;
	double x_splash = 3.0;
	double y_splash = 3.0;
	for (int i = 1; i < imax - 1; i++) {
		for (int j = 1; j < jmax - 1; j++) {
			double x = dx * i;
			double y = dy * j;

			double dist = sqrt(pow(x - x_splash, 2.0) + pow(y - y_splash, 2.0));

			if (dist < r_splash) {
				double h = 5.0 * (cos(dist / r_splash * M_PI) + 1.0);

				//original
				//grid_2d[i][j] = h;
				//old_grid_2d[i][j] = h;

				// using 1d arrays as is required 
				// for data scatter using MPI_Scatterv
				grid_1d[i * jmax + j] = h;
				old_grid_1d[i * jmax + j] = h;

			}
		}
	}
}

// used to check/validate results
// prints array to screen
void output_array(double*& array, int local_rows, int local_cols, int id)
{
	cout << "Printing processor " << id << " array: " << endl;
	for (int i = 0; i < local_rows; i++) {
		for (int j = 0; j < local_cols; j++) {
			cout << " " << array[i * local_cols + j] << " ";
		}
		cout << endl;
	}
}
// 2d version to check/validate grid
void output_array_2d(double**& array, int local_rows, int local_cols, int id) {
	cout << "Printing processor " << id << " array: " << endl;
	for (int i = 0; i < local_rows; i++) {
		for (int j = 0; j < local_cols; j++) {
			cout << " " << array[i][j] << " ";
		}
		cout << endl;
	}
}
// Boundary condition - neumann BC
// outer columns and rows are set to the adjacent(one cell inner) grid point
// ensuring a 0 gradient condition
// Corners also need to also be set.
void apply_neumann(double**& grid, int& id, int& id_row, int& local_rows, int& local_cols, int& id_col, int& m, int& n)
{
	// top row
	if (id_row == 0) {
		//top left corner
		if (id_col == 0) {
			grid[0][0] = grid[1][1];
		}
		//top right corner
		if (id_col == n - 1) {
			grid[0][local_cols + 1] = grid[1][local_cols];
		}
		//assigning values of remaining top row
		if (!((!(id_row == 0) && (id_col == 0)) && !(id_row == 0) && (id_col == n - 1))) {
			for (int j = 1; j < local_cols + 1; j++) {
				grid[0][j] = grid[1][j];
			}
		}
	}
	// bottom row
	if (id_row == m - 1) {
		// bottom left corner
		if (id_col == 0) {
			grid[local_rows + 1][0] = grid[local_rows][1];
		}
		// bottom right corner
		if (id_col == n - 1) {
			grid[local_rows + 1][local_cols + 1] = grid[local_rows][local_cols];
		}
		//assigning values of remaining bottom row
		if (!(!((id_row == m - 1) && (id_col == 0)) && !((id_row == m - 1) && (id_col == n - 1)))) {
			for (int j = 1; j < local_cols + 1; j++) {
				grid[local_rows + 1][j] = grid[local_rows][j];
			}
		}
	}
	//assigning remaining values of left column
	if ((id_col + n) % n == 0) {
		for (int i = 1; i < local_rows + 1; i++) {
			grid[i][0] = grid[i][1];
		}
	}
	//assigning remaining values of right column
	if ((id_col + 1 + n) % n == 0) {
		for (int i = 1; i < local_rows + 1; i++) {
			grid[i][local_cols + 1] = grid[i][local_cols];
		}
	}
}

// Boundary Condition - dirichlet BC
// outer cols/rows are set to 0.
// alongside looping through rows and columns, need to ensure corner/edges
// are also set to 0.
void apply_dirichlet(double**& grid, int& id, int& id_row, int& local_rows, int& local_cols, int& id_col, int& m, int& n)
{
	// top row
	if (id_row == 0) {
		//top left corner
		if (id_col == 0) {
			grid[0][0] = 0.;
		}
		//top right corner
		if (id_col == n - 1) {
			grid[0][local_cols + 1] = 0.;
		}
		//assigning values of remaining top row
		if (!((!(id_row == 0) && (id_col == 0)) && !(id_row == 0) && (id_col == n - 1))) {
			for (int j = 1; j < local_cols + 1; j++) {
				grid[0][j] = 0.;
			}
		}
	}
	// bottom row
	if (id_row == m - 1) {
		// bottom left corner
		if (id_col == 0) {
			grid[local_rows + 1][0] = 0.;
		}
		// bottom right corner
		if (id_col == n - 1) {
			grid[local_rows + 1][local_cols + 1] = 0.;
		}
		//assigning values of remaining bottom row
		if (!(!((id_row == m - 1) && (id_col == 0)) && !((id_row == m - 1) && (id_col == n - 1)))) {
			for (int j = 1; j < local_cols + 1; j++) {
				grid[local_rows + 1][j] = 0.;
			}
		}
	}
	//assigning remaining values of left column
	if ((id_col + n) % n == 0) {
		for (int i = 1; i < local_rows + 1; i++) {
			grid[i][0] = 0.;
		}
	}
	//assigning remaining values of right column
	if ((id_col + 1 + n) % n == 0) {
		for (int i = 1; i < local_rows + 1; i++) {
			grid[i][local_cols + 1] = 0.;
		}
	}
}

// converting id to index
void id_to_index(int id, int& id_row, int& id_col) {
	id_col = int(id % n);
	id_row = int(id / n);
	//if (id == 0)
		//cout << "id_col is: " << id_col << " + id_row is: " << id_row;
}
// finding id from index
int id_from_index(int id_row, int id_column, int rows, int cols) {
	return id_row * cols + id_column;
}

void find_dimensions(int p, int& rows, int& columns)		//A bit brute force - this can definitely be made more efficient!
{
	m = 1;
	n = p; //if p is a prime number this is the only way
	for (int i = 2; i < p; i++) {
		//if p is exactly divisible by i -> i is a valid size length for the grid
		if (p % i == 0)
		{
			//if the gap between the new potential factors are smaller than the current factors/arrangement
			//then use the new factors
			if (abs(i - p / i) < (abs(m - n)))
			{
				m = i; //rows
				n = p / i; //columns
			}
		}
	}
	//if (id == 0)
		//cout << "ID: " << id << " ";
		//cout << "Divide " << p << " into " << m << " by " << n << " grid" << endl;
}

// mapping i,j indices into correct datatype
// RIght, top, left, bottom
void indexing(int& i, int& j, int& index)
{
	if (i + j == 1)
	{
		if (i == 0) {
			index = 0;
		}
		if (j == 0) {
			index = 2;
		}
	}
	else if ((i + j) == -1)
	{
		if (i == 0) {
			index = 1;
		}
		if (j == 0) {
			index = 3;
		}
	}
}

// swap function
// allocates the grid -> old, new -> grid and thereby leaving new_grid 
// to be overwritten during the do_iteration
void swap(double**& domain, double**& old_domain, double**& new_domain, int local_rows, int local_cols) {
	// Swaps the two input arrays (domain and new_domain)
	double** temp;
	temp = new double* [local_rows + 2];
	for (int i = 0; i < local_rows + 2; i++) {
		temp[i] = new double[local_cols + 2];
		for (int j = 0; j < local_cols + 2; j++) {
			temp[i][j] = domain[i][j];
			domain[i][j] = new_domain[i][j];
			new_domain[i][j] = old_domain[i][j];
			old_domain[i][j] = temp[i][j];
		}
	}
	for (int j = 0; j < local_rows + 2; j++) {
		delete[] temp[j];
	}
	delete[] temp;
}

//void swap(double*** r, double*** s)
//{
//	double** pSwap = *r;
//	*r = *s;
//	*s = pSwap;
//}

// wave equation implementation carried altered for use of arrays
void do_iteration(double**& grid_2d, double** new_grid_2d, double** old_grid_2d, int local_rows, int local_cols)
{
	//boundary wall located at i = 0, i = imax-1, j = 0, j = jmax-1
	for (int i = 1; i < local_rows + 1; i++) {
		for (int j = 1; j < local_cols + 1; j++) {
			//calculation to be done in the main fluid
			new_grid_2d[i][j] = pow(dt * c, 2.0) * ((grid_2d[i + 1][j] - 2.0 * grid_2d[i][j] + grid_2d[i - 1][j]) / pow(dx, 2.0) + (grid_2d[i][j + 1] - 2.0 * grid_2d[i][j] + grid_2d[i][j - 1]) / pow(dy, 2.0)) + 2.0 * grid_2d[i][j] - old_grid_2d[i][j];
		}
	}

	t += dt;
	//cout << t << endl;
	//pointer reference swap
	//swap(&old_grid_2d, &new_grid_2d);
	//swap(&old_grid_2d, &grid_2d);

	swap(grid_2d, old_grid_2d, new_grid_2d, local_rows, local_cols);
}

void send_receive(mpi_class*& data, int& id_row, int& id_col, int& id, int& tag_num, MPI_Request*& request, int rows, int cols) {
	// Send and receive data between processes

	int id_dom = 0;
	int index_i = 0;
	int index_j = 0;
	int index = 0;
	int i_dom = 0;
	int j_dom = 0;

	int cnt = 0;
	// Iterate through neighbours of each cell
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			// only sending and receiving the four points on the stencil
			if ((i != j) && (i + j != 0)) {
				i_dom = (id_row + rows + i) % rows;
				j_dom = (id_col + cols + j) % cols;
				// finding corresponding communication id of processor for communication
				id_dom = id_from_index(i_dom, j_dom, rows, cols);
				if (id_dom < p) {
					index_i = i;
					index_j = j;
					// corresponding datatype index
					indexing(index_i, index_j, index);

					// sends and receives
					MPI_Irecv(MPI_BOTTOM, 1, data->typelist_recv_centre[index], id_dom, tag_num, MPI_COMM_WORLD, &request[cnt]);
					cnt++;
					MPI_Isend(MPI_BOTTOM, 1, data->typelist_send_centre[index], id_dom, tag_num, MPI_COMM_WORLD, &request[cnt]);
					cnt++;
				}
			}
			else {
				continue;
			}
		}
	}
	MPI_Waitall(cnt, request, MPI_STATUS_IGNORE); // Used for ensuring communication happens correctly
}

void grid_to_file(double**& grid_2d, int id, int local_rows, int local_cols, int out_cnt)
{
	// Print all data into a file
	ofstream myfile;
	std::string s0("p_" + std::to_string(id) + "_output.txt");
	myfile.open(s0, ios::app);
	for (int i = 0; i < local_rows + 2; i++) {
		for (int j = 0; j < local_cols + 2; j++) {
			myfile << "," << grid_2d[i][j];
		}
	}
	myfile << "/it/";
	myfile.close();
}

// allocate information to details file about the process, separated by "_"
void proc_details(int id, int local_rows, int local_cols, int p, int imax, int jmax, int m, int n, int id_row, int id_col) {
	ofstream myfile;
	std::string s0("p_" + std::to_string(id) + "_details.txt");
	myfile.open(s0, ios::trunc);
	myfile << id << "_" << local_rows << "_" << local_cols << "_" << p << "_"
		<< imax << "_" << jmax << "_" << m << "_" << n << "_" << id_row << "_" << id_col;

	myfile.close();
}
// used to output timings on the hpc
void mpi_timings(int id, double elapsed_time)
{
	ofstream file;
	std::string s0("p_" + std::to_string(id) + "_timings.txt");
	file.open(s0, ios::trunc);
	//file << id;
	//file << "_" << elapsed_time;
	file << elapsed_time;
}

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	imax = 100;
	jmax = 100;
	dx = x_max / ((double)imax - 1);
	dy = y_max / ((double)jmax - 1);
	t = 0.0;

	//adhering to CFL rule 
	dt = 0.1 * min(dx, dy) / c; //choosing value (0.1, 0.2, 0.3) -> is a matter of accuracy vs speed -> 0.2/0.3 may be plausible?
	int out_cnt = 0, it = 0;

	// start timing from here
	MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();

	// Assign subdomain to each process
	if (p == 1) {
		m = 1;
		n = 1;
	}
	else {
		find_dimensions(p, m, n);
	}
	// row and col position of index
	id_to_index(id, id_row, id_col);

	int rows_remainder = imax % m;
	int cols_remainder = jmax % n;

	bool flag = false;
	//find local_cols
	for (int i = n - 1; i < m * n; i = i + n) {
		if (id == i)
		{ // uneven processor
			local_cols = jmax / n + cols_remainder; //found our cols of uneven processor
			flag = true;
		}
		else
		{ // even processors
			if (!flag) {
				local_cols = jmax / n;
			}
		}
	}
	flag = false;
	//find local_rows
	for (int i = m * n - n; i <= m * n; i++) {
		if (id == i) // uneven number of procs
		{
			local_rows = imax / m + rows_remainder;
			//if (id == 0)
			//	cout << local_rows << endl;
			flag = true;
		}
		else
		{// even number of procs
			if (!flag) {
				local_rows = imax / m;
				//if (id == 0)
				//	cout << local_rows << endl;
			}
		}
	}

	proc_details(id, local_rows, local_cols, p, imax, jmax, m, n, id_row, id_col);

	// setup continuous grid
	double** grid_2d, * grid_1d;
	double** new_grid_2d, * new_grid_1d;
	double** old_grid_2d, * old_grid_1d;

	setup_continuous_array(grid_2d, grid_1d, imax, jmax);
	setup_continuous_array(new_grid_2d, new_grid_1d, imax, jmax);
	setup_continuous_array(old_grid_2d, old_grid_1d, imax, jmax);

	// applying initial disturbance only to id 0
	if (id == 0)
	{
		initial_disturbance(grid_1d, old_grid_1d, imax, jmax);
		//output_array(grid_1d, imax, jmax, id);
		//cout << endl << imax << " " << jmax << endl;
	}

	// sub-domains set up
	double** sub_grid_2d, * sub_grid_1d;
	double** sub_new_grid_2d, * sub_new_grid_1d;
	double** sub_old_grid_2d, * sub_old_grid_1d;

	setup_continuous_array(sub_grid_2d, sub_grid_1d, local_rows, local_cols);
	setup_continuous_array(sub_new_grid_2d, sub_new_grid_1d, local_rows, local_cols);
	setup_continuous_array(sub_old_grid_2d, sub_old_grid_1d, local_rows, local_cols);

	// partitioning matrix into blocks of data
	MPI_Datatype blocktype1;
	MPI_Datatype blocktype2;

	MPI_Type_vector(local_rows, local_cols, jmax, MPI_DOUBLE, &blocktype2);
	MPI_Type_create_resized(blocktype2, 0, sizeof(double), &blocktype1);
	MPI_Type_commit(&blocktype1);

	int* displacements = new int[m * n];
	int* counts = new int[m * n];

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			displacements[i * n + j] = i * jmax * local_rows + j * local_cols;
			counts[i * n + j] = 1;
		}
	}

	// MPI_Scatterv version of MPI_Scatter which dispactches blocks of data to all processors 
	// using scatterv enables varying the number of elements that is 
	// scattered from the root (process 0)
	// - takes in 1D array
	MPI_Scatterv(grid_1d, counts, displacements, blocktype1, sub_grid_1d, local_rows * local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(old_grid_1d, counts, displacements, blocktype1, sub_old_grid_1d, local_rows * local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(new_grid_1d, counts, displacements, blocktype1, sub_new_grid_1d, local_rows * local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//if (id == 0) { //testing to see what is received by each proc
	//	output_array(sub_grid_1d, local_rows, local_cols, id);
	//	cout << endl << local_rows << " " << local_cols << endl;
	//}

	// padding sub-domain grid for each - current, new, old grids
	// setup in each proc with relevant sub-domain dimensions
	double** padded_grid_2d, * padded_grid_1d;
	double** padded_new_grid_2d, * padded_new_grid_1d;
	double** padded_old_grid_2d, * padded_old_grid_1d;

	// creating padded (zero) arrays
	setup_continuous_array(padded_grid_2d, padded_grid_1d, local_rows + 2, local_cols + 2);
	setup_continuous_array(padded_new_grid_2d, padded_new_grid_1d, local_rows + 2, local_cols + 2);
	setup_continuous_array(padded_old_grid_2d, padded_old_grid_1d, local_rows + 2, local_cols + 2);

	// filling padded grid with relevant values in central region
	// from the scattered data existing in grid_1d 
	padding_grid(sub_grid_1d, padded_grid_1d, padded_grid_2d, local_rows, local_cols);
	padding_grid(sub_new_grid_1d, padded_new_grid_1d, padded_grid_2d, local_rows, local_cols);
	padding_grid(sub_old_grid_1d, padded_old_grid_1d, padded_grid_2d, local_rows, local_cols);

	// setting up the sends and receivEs:
	mpi_class* data = new mpi_class(local_rows, local_cols, padded_grid_2d);
	// Initialise data types
	data->mpi_sendtype();
	data->mpi_recvtype();
	// 8 requests per processor (4 send and 4 receive) 
	MPI_Request* request = new MPI_Request[4 * 2 * p];

	while (t < t_max)
	{
		if (boundary == "periodic")
		{
			//do an iteration
			do_iteration(data->grid, padded_new_grid_2d, padded_old_grid_2d, local_rows, local_cols);
		}
		else if (boundary == "dirichlet")
		{
			// setting boundaries before iter
			apply_dirichlet(data->grid, id, id_row, local_rows, local_cols, id_col, m, n);
			do_iteration(data->grid, padded_new_grid_2d, padded_old_grid_2d, local_rows, local_cols);
			// resetting boundaries again after iter
			apply_dirichlet(data->grid, id, id_row, local_rows, local_cols, id_col, m, n);
		}
		else if (boundary == "neumann")
		{
			// setting boundaries before iter
			apply_neumann(data->grid, id, id_row, local_rows, local_cols, id_col, m, n);
			do_iteration(data->grid, padded_new_grid_2d, padded_old_grid_2d, local_rows, local_cols);
			// resetting boundaries again after iter
			apply_neumann(data->grid, id, id_row, local_rows, local_cols, id_col, m, n);
		}

		// performing a round of communications between processes
		send_receive(data, id_row, id_col, id, tag_num, request, m, n);
		// MPI_Barrer to make sure all comms are completed
		MPI_Barrier(MPI_COMM_WORLD);

		// outputting grid to the relevant proc .txt file
		if (t_out <= t)
		{
			out_cnt++;
			grid_to_file(data->grid, id, local_rows, local_cols, out_cnt);
			t_out += dt_out;
		}
		it++;
	}
	// Barrier ensuring all proc comms have completed
	// before taking end timing reading
	MPI_Barrier(MPI_COMM_WORLD);
	double end = MPI_Wtime();
	double elapsed_time;
	if (id == 0)
	{
		cout << "The process took " << end - start << " seconds to run." << std::endl;
		elapsed_time = end - start;
		// used to print out timings to a .txt file for HPC analysis
		mpi_timings(id, elapsed_time);
	}

	// deletes preventing memory leak

	delete[] grid_1d;
	delete[] new_grid_1d;
	delete[] old_grid_1d;
	delete[] sub_grid_1d;
	delete[] sub_new_grid_1d;
	delete[] sub_old_grid_1d;
	delete[] padded_grid_1d;
	delete[] padded_new_grid_1d;
	delete[] padded_old_grid_1d;

	free_2d_array(grid_2d, jmax);
	free_2d_array(new_grid_2d, jmax);
	free_2d_array(old_grid_2d, jmax);

	free_2d_array(sub_grid_2d, local_rows);
	free_2d_array(sub_new_grid_2d, local_rows);
	free_2d_array(sub_old_grid_2d, local_rows);

	free_2d_array(padded_grid_2d, local_rows + 2);
	free_2d_array(padded_new_grid_2d, local_rows + 2);
	free_2d_array(padded_old_grid_2d, local_rows + 2);

	delete data;
	delete[] request;

	MPI_Finalize();
	return 0;
}