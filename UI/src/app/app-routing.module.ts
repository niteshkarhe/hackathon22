import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { FormTypeComponent } from './components/form-type/form-type.component';
import { HomeComponent } from './components/home/home.component';
import { JsonTypeComponent } from './components/json-type/json-type.component';
import { LoginComponent } from './components/login/login.component';

const routes: Routes = [
{
  path:'',
  component:HomeComponent,
  pathMatch:'full'
},
{
  path:'login',
  component:LoginComponent,
  pathMatch:'full'
},
{
  path:'dashboard',
  component:DashboardComponent,
  pathMatch:'full'
},
{
  path:'form-type',
  component:FormTypeComponent,
  pathMatch:'full'
},
{
  path:'json-type',
  component:JsonTypeComponent,
  pathMatch:'full'
}

];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
