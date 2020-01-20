#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <math.h>
#include <map>

using namespace std;

class NaiveB
{
    private:
        //metadata for each state
        struct metadata
        {
            int column = NULL;
            int overall_freq = 0;
            double overall_prob = 0;
            double stateP = NULL;
        };   

        //likelihood of the edge X(i) -> Y(i)
        struct likelihood
        {
            int y_freq = 0;
            double y_prob = 0;
            double likelihood = 0;
        };

        vector <vector<string> > data; // vector that holds the dataset

        map <string, metadata*> Features; // X
        map <string, metadata*> Y;  // Y or outcomes

        map <string, map<string, likelihood*> > connections;    //use to store X(i) -> Y(i) connections
      
        int data_size;  // variable that holds the size of dataset
    
        void getXY()
        {
            /* Method for Putting in features and outcomes into Features and Y Maps */

            for (auto &row : data)
            {
                for (int col = 0; col < row.size()-1; col ++)
                {
                    // if the feature is not in the map
                    if (Features.find(row[col]) == Features.end())
                    {
                        metadata* new_entryX = new metadata;
                        map<string, likelihood*> edge;

                        new_entryX->overall_freq += 1;
                        new_entryX->column = col;
 
                        Features[row[col]] = new_entryX;
                        connections[row[col]] = edge;
                        cout << "New X: " << row[col] << endl;
                    }
                    // if the feature is already in the map, update its count
                    else
                    {
                        Features[row[col]]->overall_freq += 1;
                    }
                }
                // if the outcome is not in the map
                if (Y.find(row[row.size()-1]) == Y.end())
                {
                    metadata* new_entryY = new metadata;
                    new_entryY->overall_freq += 1;
                    Y[row[row.size()-1]] = new_entryY;
                    cout << "New Y: " << row[row.size()-1] << endl;
                }
                // if outcome exists, update its count
                else
                {
                    Y[row[row.size()-1]]->overall_freq += 1;
                }
            }
        }

        void updateGProb()
        {
            /* A method for updating X and Y global probability; Target(i)'s freq / data size */
            //updating individual X global Probability
            for (auto &feature : Features)
            {
                feature.second->overall_prob = double(feature.second->overall_freq) / double(data_size);
            }

            //update individual Y global Probability
            for (auto &y : Y)
            {
                y.second->overall_prob = double(y.second->overall_freq) / double(data_size);
            }

        }

        void updateC()
        {
            //update connection between Feature and Y
            for (auto &con : connections)
            {
                for (auto &y : Y )
                {
                    for (auto &row : data)
                    {
                        // y exists in the current row of data and the feature exists in that row
                        if (y.first == row[row.size()-1] && row[Features[con.first]->column] == con.first)
                        {
                            // add edge to the connections map and update its freq
                            if (con.second.find(y.first) == con.second.end())
                            {
                                likelihood* item = new likelihood();
                                item->y_freq += 1;
                                con.second[y.first] = item;
                            }
                            // if already exists, simply update its freq
                            else
                            {
                                con.second[y.first]->y_freq += 1;
                            }
                        }
                        // if feature does not match with the y value in the row
                        else if (y.first != row[row.size()-1] && row[Features[con.first]->column] == con.first)
                        {
                            //simply add the edge to the connection, Map(X(current))[y(current)] = item
                            if (con.second.find(y.first) == con.second.end())
                            {
                                likelihood* item = new likelihood();
                                con.second[y.first] = item;
                            }
                        }
                    }

                    // update edge probability
                    if (con.second[y.first]->y_freq != 0 && Features[con.first]->overall_freq != 0)
                    {
                        con.second[y.first]->y_prob = double(con.second[y.first]->y_freq) / double(Features[con.first]->overall_freq);
                    }
                    else 
                    {
                        con.second[y.first]->y_prob = 0.0;
                    }
                }
            } 
        }

        void insert(const vector<string> &item)
        {   
            data.push_back(item);
        }

        void load(string filename)
        {
            // load file onto the vector data
            ifstream file(filename);
            string line;

            while (getline(file, line))
            {
                vector<string> item;
                istringstream input(line);
                //split the line by space
                for (string temp; input >> temp;)
                    //append each string onto the current vector
                    item.push_back(temp);
                
                //insert the row onto the data
                insert(item);
            }
        }   

    public:     
        NaiveB(string filename) 
        {
            load(filename);
            data_size = data.size();
            getXY(); updateGProb(); updateC();
        }

        void displayData()
        {
            //display the data vector
            for (auto &entry: data)
            {
                for (auto &item : entry)
                {
                    cout << item << " ";
                }
                cout << "\n";
            }
        }

        void displayTable()
        {
            /* A method for displaying the three maps; Features, Y, connections */
            cout << "Displaying X " << endl;
            for (auto &feature: Features)
            {
                cout << "Feature: " << feature.first << " occurences: " << feature.second->overall_freq << " col: " << feature.second->column << " prob: " << feature.second->overall_prob << endl;
            }

            cout << "\nDisplaying Y " << endl;
            for (auto &y: Y)
            {
                cout << "Y: " << y.first << " occurences: " << y.second->overall_freq << " prob: " << y.second->overall_prob << endl;
            }

            cout << "\nDisplaying Connections " << endl;
            for (auto &con : connections)
            {
                for (auto &edge : con.second)
                {
                    cout << "Feature: " << con.first << " to Y: " << edge.first << " ocurrences: " << edge.second->y_freq << " prob: " << edge.second->y_prob << endl;
                }
            }

        }

        void predict(string outlook, string temperature, string humidity, string wind)
        {
            // Prediction Method
            for (auto &y : Y)
            {
                //Top equation P(X(i) | Y) * P(Y)
                double top_equation = connections[outlook][y.first]->y_prob * connections[temperature][y.first]->y_prob
                                    * connections[humidity][y.first]->y_prob * connections[wind][y.first]->y_prob 
                                    * y.second->overall_prob;
                
                //P(X(i) | Y) * P(Y) / (P(X(i)))
                y.second->stateP = double(top_equation); 

                cout << "Predicting Y: " << y.first << " based on given input; probability of Y: " << y.second->stateP << endl;                
            }

        }
     
};

int main()
{
    NaiveB classifier = NaiveB("dataset.txt");
    //classifier.displayTable();
    classifier.predict("Sunny", "Hot", "Normal", "False");

}
