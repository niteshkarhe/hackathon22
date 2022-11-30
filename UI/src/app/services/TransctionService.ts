
import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Transaction } from './Transaction';
import { Observable, throwError } from 'rxjs';
import { retry, catchError } from 'rxjs/operators';
@Injectable({
  providedIn: 'root',
})
export class RestApiService {
  // Define API
  apiURL = 'localhost:5000';
  constructor(private http: HttpClient) {}
  /*========================================
    CRUD Methods for consuming RESTful API
  =========================================*/
  // Http Options
  httpOptions = {
    headers: new HttpHeaders({
      'Content-Type': 'application/json',
    }),
  };
  // HttpClient API get() method => Fetch Transactions list
  getTransactions(): Observable<Transaction> {
    return this.http
      .get<Transaction>(this.apiURL + '/Transactions')
      .pipe(retry(1), catchError(this.handleError));
  }
  // HttpClient API get() method => Fetch Transaction
  getTransaction(id: any): Observable<Transaction> {
    return this.http
      .get<Transaction>(this.apiURL + '/Transactions/' + id)
      .pipe(retry(1), catchError(this.handleError));
  }
  // HttpClient API post() method => Create Transaction
  createTransaction(transaction: any): Observable<Transaction> {

    console.debug("createTransaction called................."+ transaction);
    return this.http
      .post<Transaction>(
        this.apiURL + '/fraud/predict',
        JSON.stringify(transaction),
        this.httpOptions
      )
      .pipe(retry(1), catchError(this.handleError));


  }
  // HttpClient API put() method => Update Transaction
  updateTransaction(id: any, Transaction: any): Observable<Transaction> {
    return this.http
      .put<Transaction>(
        this.apiURL + '/Transactions/' + id,
        JSON.stringify(Transaction),
        this.httpOptions
      )
      .pipe(retry(1), catchError(this.handleError));
  }
  // HttpClient API delete() method => Delete Transaction
  deleteTransaction(id: any) {
    return this.http
      .delete<Transaction>(this.apiURL + '/Transactions/' + id, this.httpOptions)
      .pipe(retry(1), catchError(this.handleError));
  }
  // Error handling
  handleError(error: any) {
    let errorMessage = '';
    if (error.error instanceof ErrorEvent) {
      // Get client-side error
      errorMessage = error.error.message;
    } else {
      // Get server-side error
      errorMessage = `Error Code: ${error.status}\nMessage: ${error.message}`;
    }
    window.alert(errorMessage);
    return throwError(() => {
      return errorMessage;
    });
  }
}