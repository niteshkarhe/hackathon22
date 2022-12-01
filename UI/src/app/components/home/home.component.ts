import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import {RestApiService} from '../../services/TransctionService';
import {Transaction} from '../../services/Transaction';
@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})

export class HomeComponent implements OnInit {
  loading = false;
  itemsList = [];
  showForm:boolean = true;
  showJSON:boolean = false;
  list_name;
  dataSource: FraudPrevention[] = [];
  transactionJSON;


  transaction: Transaction = {
    'type': 'TRANSFER',
    'amount': 8000.00,
    'fromAccNo': "C905679615",
    'destAccNo': "CC572787442",
    'oldbalanceOrg': 10000.00
  };

  ITEMS = [{name: 1, value:'Single Transaction'}, {name: 2, value:'JSON'}];

  obj = {'type': 'Type', 'amount':'Amount', 'fromAccNo':'From Acc No', 'destAccNo':'Dest Acc No','oldbalanceOrg':'Old Balance Org','newbalanceOrg':'New Balance Org','prediction':'Prediction','indication':'Indication','Decision Tree Accuracy':'Decision Tree Accuracy','decisionTreePrediction':'Decision Tree Prediction'};
  displayedColumns = ['type', 'amount', 'fromAccNo', 'destAccNo','oldbalanceOrg','newbalanceOrg','prediction','indication','decisionTreeAccuracy','decisionTreePrediction'];
  columnsToDisplay: string[] = this.displayedColumns.slice();

  constructor(private http: HttpClient, private service : RestApiService) { }

  ngOnInit(): void {
    this.itemsList = this.ITEMS;
    this.showForm = true;
    this.showJSON = false;
    this.transactionJSON = '[{"type": "TRANSFER","amount": 18000.00,"fromAccNo": "C905679615","destAccNo": "C1023714065","oldbalanceOrg": 100000.00},{"type": "TRANSFER","amount": 8000.00,"fromAccNo": "C905679615","destAccNo": "CC572787442","oldbalanceOrg": 100000.00}]';
  }

  onSubmit(){
        this.loading = true;

    this.http.post<FraudPrevention[]>("http://localhost:5000/fraud/predict", (this.showForm)?this.transaction:JSON.parse(this.transactionJSON)).subscribe(data=> {
    this.loading = false;
      console.log("localhost:5000/fraud/predict says" + JSON.stringify(data));
      this.dataSource = data;
    },
      (error) => {                              //Error callback
        console.error(error)
        this.loading = false;
      });
    console.debug("onSubmit called................."+ this.transaction);
  }

    showFormFn(){
      this.showForm = true;
      this.showJSON = false;
    }
    showJSONFn() {
      this.showForm = false;
      this.showJSON = true;
    }

}

export interface FraudPrevention {
  type: string;
  amount: number;
  fromAccNo: string;
  destAccNo: string;
  oldbalanceOrg: number;
  newbalanceOrg: number;
  decisionTreeAccuracy: number;
  decisionTreePrediction: string,
  prediction: string,
  indication: string
}
