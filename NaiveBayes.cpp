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
        vector <vector<string> > test_data;

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

        void split_data(vector <int> &ignores, vector <vector<string> > &target)
        {
            vector <vector <string> > new_data;

            for (auto &row : target)
            {
                vector <string> entry;
                bool question = false;
                for (int col = 0; col < row.size()-1; col ++)
                {
                    if (find(ignores.begin(), ignores.end(), col) == ignores.end())
                    {
                        entry.push_back(row[col]);
                    }
                    if (row[col] == "?")
                    {
                        question = true;
                    }
                }
                if (question == false)
                    new_data.push_back(entry);
            }

            for (int row = 0; row < target.size(); row ++)
            {
                new_data[row].push_back(target[row][target[row].size()-1]);
            }

            target = new_data;
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

        const string predictY(vector<string> &inputs)
        {
            // Prediction Method
            for (auto &y : Y)
            {   
                double equation = 0.0;
                //equation P(X(i) | Y) * P(Y)
                for (auto &inp : inputs)
                {
                    if (equation ==  0.0)
                    {
                        equation = connections[inp][y.first]->y_prob;
                    }
                    else
                    {
                        equation *= connections[inp][y.first]->y_prob; 
                    }
                }
                //set  y state probability
                y.second->stateP = double(equation); 
            }
            
            //get best Y
            string best_y;
            double best_yp = 0.0;
            for (auto &y : Y)
            {
                if (best_yp < y.second->stateP)
                {
                    best_y = y.first; best_yp = y.second->stateP;
                }
            }

            return best_y;    
        }

        void load(const string filename, vector <vector<string> > &holder)
        {
            // load file onto the vector data
            ifstream file(filename);
            string line;

            while (getline(file, line))
            {
                //split the line by space
                if (!line.empty())
                {
                    
                    for (int c = 0; c < line.size(); c ++)
                    {
                        if (line[c] == ',' || line[c] == '.') line[c] = ' ';
                    }
                    
                    vector<string> item;
                    istringstream input(line);
                    for (string s; input >> s;)
                    {
                        item.push_back(s);
                    }
                    //insert the row onto the data
                    holder.push_back(item);
                }
            }
        }   

    public:     
        NaiveB(string traindata, string testdata, vector <int> &ignores) 
        {
            load(traindata, data); 
            load(testdata, test_data);
            data_size = data.size();
            split_data(ignores, data); 
            split_data(ignores, test_data);
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
    
        void predictTest()
        {
            int trues = 0;

            for (auto &row : test_data)
            {
                string current = predictY(row);

                if (current == row[row.size()-1])
                {
                    trues += 1;
                }
            }

            double accuracy = double(trues) / double(test_data.size());
            cout << "Accuracy of Model: " << accuracy << endl;

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
};

int main()
{
    vector <int> ignores;
    /*
    ignores.push_back(2);
    ignores.push_back(4);
    ignores.push_back(10);
    ignores.push_back(11);
    ignores.push_back(12);
    */
    NaiveB classifier = NaiveB("dataset.txt", "dataset.txt", ignores);
    //classifier.predictTest();
   
    
}